from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from pydantic import BaseModel
import uvicorn
from typing import Dict, Optional, List, Any
import asyncio
from enum import Enum
import torch
import threading
from queue import Queue
import time
import os
import logging
from pathlib import Path

import argparse

from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.robots.utils import make_robot, Robot
from lerobot.common.robot_devices.control_utils import control_loop, predict_action
from lerobot.common.robot_devices.control_configs import ControlPipelineConfig
from lerobot.common.policies.pretrained import PreTrainedPolicy, PreTrainedConfig
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

class InferenceStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class PolicyName(str, Enum):
    PICK = "pick"
    TRANSFER = "transfer"
    PLACE = "place"
    
    @classmethod
    def get_default_control_time(cls, policy_name):
        """Get default control time for a policy"""
        defaults = {
            cls.PICK: 15.0,      # Default 15s for pick operations
            cls.TRANSFER: 20.0,  # Default 20s for transfer operations
            cls.PLACE: 10.0      # Default 10s for place operations
        }
        return defaults.get(policy_name, 15.0)  # General default is 15s

class PolicyConfig(BaseModel):
    """Configuration for a policy"""
    path: str
    metadata_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    type: str = "act"
    fps: Optional[int] = 30
    control_time_s: Optional[float] = None

class ServerConfig(BaseModel):
    """Server configuration"""
    robot_type: str = "so100"
    policies: Dict[str, PolicyConfig] = {}
    default_task: Optional[PolicyName] = None
    control_time_s: Optional[float] = None
    display_cameras: bool = False

class InferenceRequest(BaseModel):
    task_name: PolicyName
    control_time_s: Optional[float] = None
    single_task: Optional[str] = None

class InferenceResponse(BaseModel):
    task_name: Optional[PolicyName] = None
    status: Optional[InferenceStatus] = None
    progress: int = 0
    result: Optional[Dict] = None
    error: Optional[str] = None

class PolicyListResponse(BaseModel):
    available_policies: List[PolicyName]
    default_policy: Optional[PolicyName] = None

from dataclasses import dataclass
from typing import Dict, Optional, Any

@dataclass
class TaskStatus:
    """Status buffer for a task"""
    task_name: Optional[PolicyName] = None
    status: Optional[InferenceStatus] = None
    progress: int = 0
    result: Optional[Dict] = None
    error: Optional[str] = None


class InferenceManager:
    """Manager for loading different policy weights and executing tasks"""
    def __init__(self, config: ServerConfig):
        self.config = config
        self.robot = None
        self.policies = {}
        self.current_task_status = TaskStatus()
        self.task_lock = threading.Lock()
        self.running = False
        self.task_queue = Queue()
        self.worker_thread = None
        self.logger = logging.getLogger("InferenceManager")
        
    def initialize(self):
        """Initialize the robot and load policies"""
        try:
                # Initialize robot
            self.robot = make_robot(self.config.robot_type)
            
            # Load policies
            for policy_name, policy_config in self.config.policies.items():
                self.load_policy(policy_name, policy_config)
                
            # Start worker thread
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            
            self.logger.info(f"Initialized InferenceManager with robot type {self.config.robot_type}")
            self.logger.info(f"Loaded policies: {list(self.policies.keys())}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize InferenceManager: {str(e)}")
            return False
    
    def load_policy(self, policy_name: str, policy_config: PolicyConfig):
        """Load a policy from the given configuration"""
        try:
            self.logger.info(f"Loading policy {policy_name} from {policy_config.path}")
            config = PreTrainedConfig.from_pretrained(policy_config.path)
            ds_meta = LeRobotDatasetMetadata(policy_config.metadata_path)
            policy = make_policy(config, ds_meta)
            policy.eval()
            
            self.policies[policy_name] = {
                "policy": policy,
                "config": policy_config
            }
            self.logger.info(f"Successfully loaded policy {policy_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load policy {policy_name}: {str(e)}")
            return False
    
    def get_available_policies(self) -> List[str]:
        """Get list of available policy names"""
        return list(self.policies.keys())
    
    async def create_inference_task(self, request: InferenceRequest) -> PolicyName:
        """Create a new inference task and add it to the queue"""
        with self.task_lock:
            if request.task_name.value not in self.policies:
                raise HTTPException(status_code=400, detail=f"Policy {request.task_name} not available")
            
            # Update task status
            self.current_task_status = TaskStatus(
                task_name=request.task_name,
                status=InferenceStatus.PENDING,
                progress=0
            )
            
            # Get appropriate control time (in order of precedence):
            # 1. Request-specific control_time_s if provided
            # 2. Policy-specific control_time_s from config if available
            # 3. Policy-specific default based on task type
            # 4. Server default control_time_s
            # 5. General default (15s)
            policy_config = self.policies[request.task_name.value]["config"]
            policy_control_time = policy_config.control_time_s
            
            control_time = (request.control_time_s or 
                           policy_control_time or 
                           PolicyName.get_default_control_time(request.task_name) or 
                           self.config.control_time_s or 
                           15.0)
            
            # Add task to queue
            self.task_queue.put({
                "task_name": request.task_name,
                "control_time_s": control_time,
                "single_task": request.single_task
            })
            
            return request.task_name
    
    def get_current_task_status(self) -> Dict[str, Any]:
        """Get the current task status"""
        with self.task_lock:
            return {
                "task_name": self.current_task_status.task_name,
                "status": self.current_task_status.status,
                "progress": self.current_task_status.progress,
                "result": self.current_task_status.result,
                "error": self.current_task_status.error
            }
    
    def _worker_loop(self):
        """Worker loop for processing inference tasks"""
        self.running = True
        while self.running:
            try:
                # Get task from queue
                if self.task_queue.empty():
                    time.sleep(0.1)
                    continue
                
                task = self.task_queue.get()
                task_name = task["task_name"]
                control_time_s = task["control_time_s"]
                single_task = task["single_task"]
                
                # Update task status
                with self.task_lock:
                    self.current_task_status.status = InferenceStatus.RUNNING
                    self.current_task_status.progress = 0
                
                # Get policy
                policy_data = self.policies[task_name.value]
                policy = policy_data["policy"]
                policy_config = policy_data["config"]
                
                # Execute task
                self.logger.info(f"Executing task {task_name} with policy {policy_config.path}")
                result = self._execute_task(policy, policy_config, control_time_s, single_task)
                
                # Update task status
                with self.task_lock:
                    self.current_task_status.status = InferenceStatus.COMPLETED
                    self.current_task_status.progress = 100
                    self.current_task_status.result = result
                
                self.logger.info(f"Completed task {task_name}")
                
            except Exception as e:
                self.logger.error(f"Error in worker loop: {str(e)}")
                with self.task_lock:
                    self.current_task_status.status = InferenceStatus.FAILED
                    self.current_task_status.error = str(e)
    
    def _execute_task(self, policy, policy_config, control_time_s, single_task):
        """Execute a task using the given policy"""
        try:
            # Execute control loop
            start_time = time.time()
            
            # Setup events dict for control loop
            events = {"exit_early": False}
            
            # Run control loop
            control_loop(
                robot=self.robot,
                control_time_s=control_time_s,
                teleoperate=False,
                display_cameras=self.config.display_cameras,
                events=events,
                policy=policy,
                fps=policy_config.fps,
                single_task=single_task
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            return {
                "execution_time": execution_time,
                "control_time_s": control_time_s,
                "single_task": single_task
            }
            
        except Exception as e:
            self.logger.error(f"Error executing task: {str(e)}")
            raise
    
    def shutdown(self):
        """Shutdown the inference manager"""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        if self.robot and self.robot.is_connected:
            self.robot.disconnect()

# FastAPI application
app = FastAPI(title="Robot Control API")

# Initialize inference manager
server_config = None
inference_manager = None

@app.on_event("startup")
async def startup_event():
    global server_config, inference_manager
    # Load configuration
    config_path = os.environ.get("CONFIG_PATH", "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path, "r") as f:
            config_data = json.load(f)
        server_config = ServerConfig(**config_data)
    else:
        # Default configuration
        server_config = ServerConfig(
            robot_type="so100",
            policies={
                "pick": PolicyConfig(path="/path/to/pick_policy"),
                "transfer": PolicyConfig(path="/path/to/transfer_policy"),
                "place": PolicyConfig(path="/path/to/place_policy")
            },
            default_task=PolicyName.PICK,
            control_time_s=30.0,
            display_cameras=False
        )
    
    # Initialize inference manager
    inference_manager = InferenceManager(server_config)
    if not inference_manager.initialize():
        logging.error("Failed to initialize inference manager")

@app.on_event("shutdown")
async def shutdown_event():
    global inference_manager
    if inference_manager:
        inference_manager.shutdown()

@app.post("/inference", response_model=InferenceResponse)
async def create_inference(request: InferenceRequest):
    """Create a new inference task
    
    Args:
        request: The inference request containing task name
        
    Returns:
        InferenceResponse with initial task status
    """
    task_name = await inference_manager.create_inference_task(request)
    return InferenceResponse(
        task_name=task_name,
        status=InferenceStatus.PENDING,
        progress=0
    )

@app.get("/inference/status", response_model=InferenceResponse)
async def get_current_status():
    """Get status of current/latest inference task from buffer
    
    Returns:
        InferenceResponse with current task status and progress
    """
    # Get status without blocking
    status = inference_manager.get_current_task_status()
    return InferenceResponse(**status)

@app.get("/policies", response_model=PolicyListResponse)
async def get_available_policies():
    """Get list of available policies
    
    Returns:
        List of available policy names
    """
    policies = inference_manager.get_available_policies()
    return PolicyListResponse(
        available_policies=[PolicyName(p) for p in policies],
        default_policy=server_config.default_task
    )

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Robot Control API Server")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8888, help="Port to bind the server to")
    args = parser.parse_args()
    
    # Set configuration path in environment
    os.environ["CONFIG_PATH"] = args.config
    
    # Start inference server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()