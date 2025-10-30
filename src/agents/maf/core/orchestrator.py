"""
Workflow Orchestrator for Multi-Agent Framework
"""

import asyncio
import logging
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class WorkflowStatus(Enum):
    """Status of a workflow"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionStrategy(Enum):
    """Workflow execution strategy"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONSENSUS = "consensus"
    PIPELINE = "pipeline"
    DYNAMIC = "dynamic"


@dataclass
class WorkflowStage:
    """Single stage in a workflow"""
    name: str
    agent: str
    action: str
    timeout: int = 60
    required: bool = True
    dependencies: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.config is None:
            self.config = {}


@dataclass
class WorkflowResult:
    """Result of a workflow execution"""
    workflow_id: str
    status: WorkflowStatus
    stages_completed: List[str]
    stages_failed: List[str]
    results: Dict[str, Any]
    errors: List[str]
    start_time: str
    end_time: str
    execution_time: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return {
            'workflow_id': self.workflow_id,
            'status': self.status.value,
            'stages_completed': self.stages_completed,
            'stages_failed': self.stages_failed,
            'results': self.results,
            'errors': self.errors,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'execution_time': self.execution_time,
            'metadata': self.metadata
        }


class WorkflowOrchestrator:
    """Orchestrates multi-agent workflows"""

    def __init__(self, agents: Dict[str, Any], message_bus: Any = None):
        self.agents = agents
        self.message_bus = message_bus
        self.workflows = {}
        self.active_workflows = {}

        # Logging
        self.logger = logging.getLogger('WorkflowOrchestrator')

        # Statistics
        self.total_workflows = 0
        self.completed_workflows = 0
        self.failed_workflows = 0
        self.workflow_history = []

        self.logger.info("Orchestrator initialized with %s agents", len(agents))

    def register_workflow(
        self,
        name: str,
        stages: List[WorkflowStage],
        strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    ):
        """Register a workflow definition"""

        self.workflows[name] = {
            'name': name,
            'stages': stages,
            'strategy': strategy
        }

        self.logger.info(
            "Workflow '%s' registered with %s stages (%s strategy)",
            name, len(stages), strategy.value
        )

    async def execute_workflow(
        self,
        workflow_name: str,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        strategy_override: Optional[ExecutionStrategy] = None
    ) -> WorkflowResult:
        """Execute a workflow"""

        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")

        # Generate workflow ID
        workflow_id = str(uuid.uuid4())

        # Get workflow definition
        workflow_def = self.workflows[workflow_name]
        stages = workflow_def['stages']
        strategy = strategy_override or workflow_def['strategy']

        # Initialize workflow state
        workflow_state = {
            'id': workflow_id,
            'name': workflow_name,
            'status': WorkflowStatus.RUNNING,
            'stages_completed': [],
            'stages_failed': [],
            'results': {},
            'errors': [],
            'context': context or {},
            'input_data': input_data,
            'start_time': datetime.now(timezone.utc).isoformat()
        }

        self.active_workflows[workflow_id] = workflow_state
        self.total_workflows += 1

        self.logger.info(
            "[%s] Starting workflow '%s' with strategy %s",
            workflow_id, workflow_name, strategy.value
        )

        # Broadcast workflow start
        if self.message_bus:
            await self.message_bus.publish({
                'type': 'workflow_started',
                'sender': 'orchestrator',
                'content': {
                    'workflow_id': workflow_id,
                    'workflow_name': workflow_name,
                    'strategy': strategy.value
                }
            })

        start_time = time.time()

        try:
            # Execute based on strategy
            if strategy == ExecutionStrategy.SEQUENTIAL:
                result = await self._execute_sequential(workflow_id, stages, workflow_state)
            elif strategy == ExecutionStrategy.PARALLEL:
                result = await self._execute_parallel(workflow_id, stages, workflow_state)
            elif strategy == ExecutionStrategy.CONSENSUS:
                result = await self._execute_consensus(workflow_id, stages, workflow_state)
            elif strategy == ExecutionStrategy.PIPELINE:
                result = await self._execute_pipeline(workflow_id, stages, workflow_state)
            elif strategy == ExecutionStrategy.DYNAMIC:
                result = await self._execute_dynamic(workflow_id, stages, workflow_state)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            workflow_state['status'] = WorkflowStatus.COMPLETED
            self.completed_workflows += 1

            self.logger.info("[%s] Workflow completed successfully", workflow_id)

        except Exception as e:
            workflow_state['status'] = WorkflowStatus.FAILED
            workflow_state['errors'].append(str(e))
            self.failed_workflows += 1

            self.logger.error("[%s] Workflow failed: %s", workflow_id, e, exc_info=True)

        finally:
            # Finalize workflow
            execution_time = time.time() - start_time
            workflow_state['end_time'] = datetime.now(timezone.utc).isoformat()

            # Create result
            result = WorkflowResult(
                workflow_id=workflow_id,
                status=workflow_state['status'],
                stages_completed=workflow_state['stages_completed'],
                stages_failed=workflow_state['stages_failed'],
                results=workflow_state['results'],
                errors=workflow_state['errors'],
                start_time=workflow_state['start_time'],
                end_time=workflow_state['end_time'],
                execution_time=execution_time,
                metadata={
                    'workflow_name': workflow_name,
                    'strategy': strategy.value
                }
            )

            # Store in history
            self.workflow_history.append(result.to_dict())

            # Clean up active workflow
            del self.active_workflows[workflow_id]

            # Broadcast workflow completion
            if self.message_bus:
                await self.message_bus.publish({
                    'type': 'workflow_completed',
                    'sender': 'orchestrator',
                    'content': result.to_dict()
                })

            return result

    async def _execute_sequential(
        self, workflow_id: str, stages: List[WorkflowStage], state: Dict
    ) -> Dict:
        """Execute stages sequentially"""

        self.logger.debug("[%s] Executing sequential workflow", workflow_id)

        current_context = {
            'workflow_id': workflow_id,
            'input_data': state['input_data'],
            'previous_results': {}
        }

        for stage in stages:
            self.logger.info("[%s] Executing stage: %s", workflow_id, stage.name)

            # Check dependencies
            if not self._check_dependencies(stage, state['stages_completed']):
                self.logger.warning(
                    "[%s] Skipping stage %s - dependencies not met",
                    workflow_id, stage.name
                )
                continue

            try:
                # Execute stage
                result = await self._execute_stage(workflow_id, stage, current_context)

                # Update state
                state['stages_completed'].append(stage.name)
                state['results'][stage.name] = result

                # Update context for next stage
                current_context['previous_results'][stage.name] = result

                self.logger.info("[%s] Stage %s completed successfully", workflow_id, stage.name)

            except Exception as e:
                state['stages_failed'].append(stage.name)
                state['errors'].append(f"Stage {stage.name} failed: {str(e)}")

                self.logger.error("[%s] Stage %s failed: %s", workflow_id, stage.name, e)

                if stage.required:
                    raise Exception(f"Required stage {stage.name} failed")

        return state['results']

    async def _execute_parallel(
        self, workflow_id: str, stages: List[WorkflowStage], state: Dict
    ) -> Dict:
        """Execute stages in parallel"""

        self.logger.debug("[%s] Executing parallel workflow", workflow_id)

        context = {
            'workflow_id': workflow_id,
            'input_data': state['input_data']
        }

        # Group stages by dependencies
        stage_groups = self._group_stages_by_dependencies(stages)

        for group in stage_groups:
            self.logger.info(
                "[%s] Executing parallel group with %s stages",
                workflow_id, len(group)
            )

            # Execute group in parallel
            tasks = []
            for stage in group:
                task = self._execute_stage_safe(workflow_id, stage, context.copy())
                tasks.append(task)

            # Wait for all tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for stage, result in zip(group, results):
                if isinstance(result, Exception):
                    state['stages_failed'].append(stage.name)
                    state['errors'].append(f"Stage {stage.name} failed: {str(result)}")

                    if stage.required:
                        raise Exception(f"Required stage {stage.name} failed")
                else:
                    state['stages_completed'].append(stage.name)
                    state['results'][stage.name] = result
                    context[stage.name] = result

        return state['results']

    async def _execute_consensus(
        self, workflow_id: str, stages: List[WorkflowStage], state: Dict
    ) -> Dict:
        """Execute stages and require consensus"""

        self.logger.debug("[%s] Executing consensus workflow", workflow_id)

        context = {
            'workflow_id': workflow_id,
            'input_data': state['input_data']
        }

        # Execute all stages in parallel
        tasks = []
        for stage in stages:
            task = self._execute_stage_safe(workflow_id, stage, context.copy())
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze results for consensus
        consensus_results = {}
        for stage, result in zip(stages, results):
            if isinstance(result, Exception):
                state['stages_failed'].append(stage.name)
                state['errors'].append(f"Stage {stage.name} failed: {str(result)}")
            else:
                state['stages_completed'].append(stage.name)
                state['results'][stage.name] = result

                # Build consensus (simplified - could be more sophisticated)
                if stage.action not in consensus_results:
                    consensus_results[stage.action] = []
                consensus_results[stage.action].append(result)

        # Determine consensus
        for action, results_list in consensus_results.items():
            if len(results_list) > 1:
                # Simple majority or unanimous consensus logic
                state['results'][f"{action}_consensus"] = self._determine_consensus(results_list)

        return state['results']

    async def _execute_pipeline(
        self, workflow_id: str, stages: List[WorkflowStage], state: Dict
    ) -> Dict:
        """Execute stages as a pipeline with data flowing through"""

        self.logger.debug("[%s] Executing pipeline workflow", workflow_id)

        pipeline_data = state['input_data'].copy()

        for stage in stages:
            self.logger.info("[%s] Pipeline stage: %s", workflow_id, stage.name)

            context = {
                'workflow_id': workflow_id,
                'pipeline_data': pipeline_data,
                'stage_index': stages.index(stage),
                'total_stages': len(stages)
            }

            try:
                # Execute stage with pipeline data
                result = await self._execute_stage(workflow_id, stage, context)

                # Update pipeline data with result
                if isinstance(result, dict):
                    pipeline_data.update(result)
                else:
                    pipeline_data['stage_' + stage.name] = result

                state['stages_completed'].append(stage.name)
                state['results'][stage.name] = result

            except Exception as e:
                state['stages_failed'].append(stage.name)
                state['errors'].append(f"Pipeline stage {stage.name} failed: {str(e)}")

                if stage.required:
                    raise Exception(f"Required pipeline stage {stage.name} failed")

        state['results']['final_output'] = pipeline_data
        return state['results']

    async def _execute_dynamic(
        self, workflow_id: str, stages: List[WorkflowStage], state: Dict
    ) -> Dict:
        """Execute stages dynamically based on conditions"""

        self.logger.debug("[%s] Executing dynamic workflow", workflow_id)

        completed_stages = set()
        stage_queue = stages.copy()

        context = {
            'workflow_id': workflow_id,
            'input_data': state['input_data'],
            'results': {}
        }

        while stage_queue:
            # Find executable stages
            executable = []
            for stage in stage_queue:
                if self._check_dependencies(stage, list(completed_stages)):
                    executable.append(stage)

            if not executable:
                # No stages can be executed - might be a deadlock
                self.logger.warning(
                    "[%s] No executable stages found - possible deadlock",
                    workflow_id
                )
                break

            # Execute available stages
            for stage in executable:
                self.logger.info("[%s] Dynamically executing: %s", workflow_id, stage.name)

                try:
                    result = await self._execute_stage(workflow_id, stage, context)

                    state['stages_completed'].append(stage.name)
                    state['results'][stage.name] = result
                    context['results'][stage.name] = result
                    completed_stages.add(stage.name)

                    # Remove from queue
                    stage_queue.remove(stage)

                    # Check if result triggers new stages
                    new_stages = self._get_triggered_stages(result, stage.config)
                    if new_stages:
                        self.logger.info(
                            "[%s] Triggered %s new stages",
                            workflow_id, len(new_stages)
                        )
                        stage_queue.extend(new_stages)

                except Exception as e:
                    state['stages_failed'].append(stage.name)
                    state['errors'].append(f"Stage {stage.name} failed: {str(e)}")
                    stage_queue.remove(stage)

                    if stage.required:
                        raise Exception(f"Required stage {stage.name} failed")

        return state['results']

    async def _execute_stage(self, workflow_id: str, stage: WorkflowStage, context: Dict) -> Any:
        """Execute a single stage"""

        # Get agent
        agent = self.agents.get(stage.agent)
        if not agent:
            raise ValueError(f"Agent '{stage.agent}' not found")

        # Prepare task
        task = {
            'type': stage.action,
            'stage': stage.name,
            'config': stage.config,
            **context
        }

        # Generate task ID
        task_id = f"{workflow_id}_{stage.name}_{uuid.uuid4().hex[:8]}"

        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                agent.process(task_id, task, context),
                timeout=stage.timeout
            )

            if result.success:
                return result.result
            raise Exception(result.error or "Stage execution failed")

        except asyncio.TimeoutError:
            raise Exception(f"Stage timed out after {stage.timeout}s")

    async def _execute_stage_safe(
        self, workflow_id: str, stage: WorkflowStage, context: Dict
    ) -> Any:
        """Execute a stage safely (for parallel execution)"""
        try:
            return await self._execute_stage(workflow_id, stage, context)
        except Exception as e:
            return e

    def _check_dependencies(self, stage: WorkflowStage, completed: List[str]) -> bool:
        """Check if stage dependencies are met"""

        if not stage.dependencies:
            return True

        return all(dep in completed for dep in stage.dependencies)

    def _group_stages_by_dependencies(
        self, stages: List[WorkflowStage]
    ) -> List[List[WorkflowStage]]:
        """Group stages that can be executed in parallel"""

        groups = []
        processed = set()

        while len(processed) < len(stages):
            group = []
            for stage in stages:
                if stage.name not in processed:
                    if self._check_dependencies(stage, list(processed)):
                        group.append(stage)

            if not group:
                # Remaining stages have unmet dependencies
                break

            groups.append(group)
            for stage in group:
                processed.add(stage.name)

        return groups

    def _determine_consensus(self, results: List[Any]) -> Any:
        """Determine consensus from multiple results"""

        # Simple majority vote for now

        if all(isinstance(r, str) for r in results):
            # String consensus - most common
            counter = Counter(results)
            return counter.most_common(1)[0][0]
        elif all(isinstance(r, bool) for r in results):
            # Boolean consensus - majority
            return sum(results) > len(results) / 2
        elif all(isinstance(r, (int, float)) for r in results):
            # Numeric consensus - average
            return sum(results) / len(results)
        else:
            # Complex types - return all
            return results

    def _get_triggered_stages(self, result: Any, config: Dict) -> List[WorkflowStage]:
        """Get stages triggered by a result"""

        # This would check result against conditions and return new stages
        # For now, returning empty list
        return []

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict]:
        """Get status of an active workflow"""

        return self.active_workflows.get(workflow_id)

    async def cancel_workflow(self, workflow_id: str):
        """Cancel an active workflow"""

        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]['status'] = WorkflowStatus.CANCELLED
            self.logger.info("Workflow %s cancelled", workflow_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""

        success_rate = (
            (self.completed_workflows / self.total_workflows * 100)
            if self.total_workflows > 0 else 0
        )

        return {
            'total_workflows': self.total_workflows,
            'completed_workflows': self.completed_workflows,
            'failed_workflows': self.failed_workflows,
            'success_rate': f"{success_rate:.2f}%",
            'active_workflows': len(self.active_workflows),
            'registered_workflows': len(self.workflows)
        }
