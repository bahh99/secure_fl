"""secaggexample: A Flower with SecAgg\+ app."""

from logging import INFO, DEBUG, WARNING, ERROR
from typing import List, Tuple, Dict, Any

from secaggexample.task import get_weights, make_net

from flwr.common import Context, log, Metrics
from flwr.common.secure_aggregation.quantization import quantize
from flwr.server import Grid, LegacyContext
from flwr.server.workflow.constant import MAIN_PARAMS_RECORD
from flwr.server.workflow.secure_aggregation.secaggplus_workflow import (
    SecAggPlusWorkflow,
    WorkflowState,
)


class SecAggPlusWorkflowWithLogs(SecAggPlusWorkflow):
    """The SecAggPlusWorkflow augmented for this example.

    This class includes additional logging and modifies one of the FitIns to instruct
    the target client to simulate a dropout.
    """

    node_ids = []

    def __call__(self, grid: Grid, context: Context) -> None:
        first_3_params = get_weights(make_net())[0].flatten()[:3]
        _quantized = quantize(
            [first_3_params for _ in range(5)],
            self.clipping_range,
            self.quantization_range,
        )
        log(INFO, "")
        log(
            INFO,
            "################################ Introduction ################################",
        )
        log(
            INFO,
            "In the example, clients will skip model training and evaluation",
        )
        log(INFO, "for demonstration purposes.")
        log(
            INFO,
            "Client 0 is configured to drop out before uploading the masked vector.",
        )
        log(INFO, "After quantization, the raw vectors will look like:")
        for i in range(1, 5):
            log(INFO, "\t%s... from Client %s", _quantized[i], i)
        log(
            INFO,
            "Numbers are rounded to integers stochastically during the quantization, ",
        )
        log(INFO, "and thus vectors may not be identical.")
        log(
            INFO,
            "The above raw vectors are hidden from the ServerApp through adding masks.",
        )
        log(INFO, "")
        log(
            INFO,
            "########################## Secure Aggregation Start ##########################",
        )

        super().__call__(grid, context)

        ndarrays = context.state[MAIN_PARAMS_RECORD].to_numpy_ndarrays()
        log(
            INFO,
            "Weighted average of parameters (dequantized): %s...",
            ndarrays[0].flatten()[:3],
        )
        log(
            INFO,
            "########################### Secure Aggregation End ###########################",
        )
        log(INFO, "")

    def setup_stage(
        self, grid: Grid, context: LegacyContext, state: WorkflowState
    ) -> bool:
        ret = super().setup_stage(grid, context, state)
        self.node_ids = list(state.active_node_ids)

        if not self.node_ids:
            log(INFO, "No active nodes available for setup.")
            return ret

        if self.node_ids[0] in state.nid_to_fitins:
            state.nid_to_fitins[self.node_ids[0]]["fitins.config"]["drop"] = True
            state.nid_to_fitins[self.node_ids[0]]["fitins.config"]["attacker"] = True
        else:
            log(INFO, "Node ID not found in fitins. Skipping attacker assignment.")
        
        state.nid_to_fitins[self.node_ids[1]]["fitins.config"]["defense"] = "gaussian+clip"

        return ret

    def collect_masked_vectors_stage(
        self, grid: Grid, context: LegacyContext, state: WorkflowState
    ) -> bool:
        ret = super().collect_masked_vectors_stage(grid, context, state)

        for node_id in state.sampled_node_ids - state.active_node_ids:
            log(INFO, "Client %s dropped out.", self.node_ids.index(node_id))

        if len(state.aggregate_ndarrays) > 1:
            log(INFO, "Sum of masked parameters: %s...",
                state.aggregate_ndarrays[1].flatten()[:3])
        else:
            log(INFO, "No masked vectors collected; skipping parameter summary.")

        return ret
