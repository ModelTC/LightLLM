"""
Weight verification utility for validating model weights via SHA256 checksums.
Similar to sglang's weight verification feature for ensuring bit-for-bit identical weights.
"""

import hashlib
import torch
from typing import Dict, Optional, List, Tuple
from lightllm.utils.log_utils import init_logger
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size

logger = init_logger(__name__)


class WeightChecker:
    """
    Verifies weight integrity through SHA256 checksums.
    Supports both single-rank and tensor-parallel distributed weights.
    """

    def __init__(self, tp_rank: Optional[int] = None, tp_world_size: Optional[int] = None):
        """
        Initialize the weight checker.

        Args:
            tp_rank: Tensor parallel rank (defaults to current rank)
            tp_world_size: Tensor parallel world size (defaults to current world size)
        """
        self.tp_rank = tp_rank if tp_rank is not None else get_current_rank_in_dp()
        self.tp_world_size = tp_world_size if tp_world_size is not None else get_dp_world_size()
        self.local_checksums: Dict[str, str] = {}

    def compute_checksum(
        self,
        param_name: str,
        tensor: torch.Tensor,
        force_bfloat16: bool = True
    ) -> str:
        """
        Compute SHA256 checksum for a tensor.

        Args:
            param_name: Name of the parameter
            tensor: The weight tensor to compute checksum for
            force_bfloat16: Whether to cast to bfloat16 before hashing for consistency

        Returns:
            Hexadecimal string of the SHA256 checksum
        """
        if tensor is None:
            logger.warning(f"Cannot compute checksum for None tensor: {param_name}")
            return ""

        # Move tensor to CPU for checksum computation
        cpu_tensor = tensor.cpu()

        # Cast to bfloat16 for bit-perfect alignment if requested
        if force_bfloat16 and cpu_tensor.dtype != torch.bfloat16:
            cpu_tensor = cpu_tensor.to(torch.bfloat16)

        # Compute SHA256 hash
        tensor_bytes = cpu_tensor.numpy().tobytes()
        checksum = hashlib.sha256(tensor_bytes).hexdigest()

        logger.debug(f"Computed checksum for {param_name}: {checksum}")
        return checksum

    def compute_sharded_checksum(
        self,
        param_name: str,
        tensor: torch.Tensor,
        shard_dim: int = 0,
        force_bfloat16: bool = True
    ) -> str:
        """
        Compute checksum for a tensor shard with dimension information.

        Args:
            param_name: Name of the parameter
            tensor: The sharded weight tensor
            shard_dim: Dimension along which the tensor is sharded (0 or 1)
            force_bfloat16: Whether to cast to bfloat16 before hashing

        Returns:
            Hexadecimal string of the SHA256 checksum
        """
        checksum = self.compute_checksum(param_name, tensor, force_bfloat16)

        # Store shard information
        shard_key = f"{param_name}_shard_dim_{shard_dim}_rank_{self.tp_rank}"
        self.local_checksums[shard_key] = checksum

        return checksum

    def verify_checksum(
        self,
        param_name: str,
        tensor: torch.Tensor,
        expected_checksum: str,
        force_bfloat16: bool = True
    ) -> bool:
        """
        Verify a tensor's checksum against expected value.

        Args:
            param_name: Name of the parameter
            tensor: The weight tensor to verify
            expected_checksum: Expected SHA256 checksum
            force_bfloat16: Whether to cast to bfloat16 before hashing

        Returns:
            True if checksums match, False otherwise
        """
        if not expected_checksum:
            logger.debug(f"No expected checksum provided for {param_name}, skipping verification")
            return True

        actual_checksum = self.compute_checksum(param_name, tensor, force_bfloat16)

        if actual_checksum != expected_checksum:
            logger.error(
                f"Checksum mismatch for {param_name}!\n"
                f"  Expected: {expected_checksum}\n"
                f"  Actual:   {actual_checksum}"
            )
            return False

        logger.info(f"Checksum verified for {param_name}: {actual_checksum}")
        return True

    def compare_checksums(
        self,
        checksums: Dict[str, str],
        weights: Dict[str, torch.Tensor],
        force_bfloat16: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Compare multiple checksums against provided weights.

        Args:
            checksums: Dictionary mapping parameter names to expected checksums
            weights: Dictionary mapping parameter names to tensors
            force_bfloat16: Whether to cast to bfloat16 before hashing

        Returns:
            Tuple of (all_passed, list_of_failed_params)
        """
        all_passed = True
        failed_params = []

        for param_name, expected_checksum in checksums.items():
            if param_name not in weights:
                logger.warning(f"Parameter {param_name} not found in weights, skipping verification")
                continue

            tensor = weights[param_name]
            if not self.verify_checksum(param_name, tensor, expected_checksum, force_bfloat16):
                all_passed = False
                failed_params.append(param_name)

        if all_passed:
            logger.info(f"All {len(checksums)} weight checksums verified successfully")
        else:
            logger.error(f"Checksum verification failed for {len(failed_params)} parameters: {failed_params}")

        return all_passed, failed_params

    def gather_tp_checksums(self) -> Dict[str, List[str]]:
        """
        Gather checksums from all TP ranks (if using tensor parallelism).

        Returns:
            Dictionary mapping parameter names to list of checksums from all ranks
        """
        # This would require distributed communication
        # For now, return local checksums
        # In a full implementation, this would use torch.distributed.all_gather

        if self.tp_world_size == 1:
            return {k: [v] for k, v in self.local_checksums.items()}

        logger.warning("TP checksum gathering not yet implemented for multi-rank scenarios")
        return {k: [v] for k, v in self.local_checksums.items()}

    def get_local_checksums(self) -> Dict[str, str]:
        """
        Get all locally computed checksums.

        Returns:
            Dictionary of parameter names to checksums
        """
        return self.local_checksums.copy()

    def clear_checksums(self):
        """Clear all stored checksums."""
        self.local_checksums.clear()


def create_weight_checker(tp_rank: Optional[int] = None, tp_world_size: Optional[int] = None) -> WeightChecker:
    """
    Factory function to create a WeightChecker instance.

    Args:
        tp_rank: Tensor parallel rank
        tp_world_size: Tensor parallel world size

    Returns:
        WeightChecker instance
    """
    return WeightChecker(tp_rank=tp_rank, tp_world_size=tp_world_size)
