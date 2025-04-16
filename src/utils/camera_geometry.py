import sys

import torch
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")


def lidar_to_camera_coordinates(
    point_cloud: torch.Tensor,
    intrinsic_mat: torch.Tensor,
    extrinsic_mat: torch.Tensor,
    without_batch_dim: bool = False,
):
    """
    Project 3D LiDAR points to 3D camera coordinates using camera parameters.

    Parameters
    ----------
    point_cloud : torch.Tensor
        LiDAR point cloud with shape (B, N, 3) or (B, N, 4) where B is batch size,
        N is number of points, and the last dimension contains XYZ coordinates
        (and optionally intensity).
    intrinsic_mat : torch.Tensor
        Camera intrinsic matrix with shape (3, 3) containing focal lengths and principal point.
    extrinsic_mat : torch.Tensor
        Camera extrinsic matrix with shape (3, 4) representing the transformation
        from LiDAR coordinates to camera coordinates.
    without_batch_dim : bool, optional
        If True, remove the batch dimension from the output.

    Returns
    -------
    torch.Tensor
        Projected points in image coordinates with shape (B, N, 3), where the last dimension
        contains (x, y, z) with x, y being pixel coordinates and z being depth.
    """
    if without_batch_dim:
        if len(point_cloud.shape) == 2:
            point_cloud = point_cloud.unsqueeze(0)
        else:
            logger.error(f"Invalid point cloud shape: {point_cloud.shape}")
            raise ValueError(f"Invalid point cloud shape: {point_cloud.shape}")

    projection_mat = intrinsic_mat @ extrinsic_mat
    tmp = point_cloud.clone()
    if tmp.shape[-1] == 4:  # having the LiDAR intensity values
        tmp[:, :, 3] = 1
    elif tmp.shape[-1] == 3:
        tmp = torch.cat([tmp, torch.ones_like(tmp[:, :, :1])], dim=-1)
    else:
        raise ValueError(f"Invalid point cloud shape: {tmp.shape}")
    projected_points = projection_mat[None, :, :] @ tmp.transpose(1, 2)

    result = projected_points[:, :3, :].transpose(1, 2)

    if without_batch_dim:
        result = result.squeeze(0)

    return result


def camera_to_image_coordinates(
    point_cloud: torch.Tensor,  # [B,N,3] or [N,3]
    without_batch_dim: bool = False,
):
    """
    Project 3D camera points to 2D image coordinates.

    Parameters
    ----------
    point_cloud : torch.Tensor
        Camera points with shape (B, N, 3) where B is batch size,
        N is number of points, and the last dimension contains XYZ coordinates
        in camera space. If without_batch_dim is True, shape should be (N, 3).
    without_batch_dim : bool, optional
        If True, assumes input has no batch dimension and returns output without batch dimension.

    Returns
    -------
    torch.Tensor
        Projected points in image coordinates with shape (B, N, 3) or (N, 3) if without_batch_dim is True,
        where the last dimension contains (u, v, d) with u, v being pixel coordinates and d being depth in camera space.
    """
    if without_batch_dim:
        if len(point_cloud.shape) == 2:
            point_cloud = point_cloud.unsqueeze(0)
        else:
            logger.error(f"Invalid point cloud shape: {point_cloud.shape}")
            raise ValueError(f"Invalid point cloud shape: {point_cloud.shape}")

    D = point_cloud[:, :, 2]
    U = point_cloud[:, :, 0] / D
    V = point_cloud[:, :, 1] / D
    result = torch.stack([U, V, D], dim=-1)

    if without_batch_dim:
        result = result.squeeze(0)

    return result


def lidar_to_image_coordinates(
    point_cloud: torch.Tensor,
    intrinsic_mat: torch.Tensor,
    extrinsic_mat: torch.Tensor,
    without_batch_dim: bool = False,
):
    """
    Project 3D LiDAR points to 2D image coordinates.

    Parameters
    ----------
    point_cloud : torch.Tensor
        LiDAR point cloud with shape (B, N, 3) or (N, 3) if without_batch_dim is True.
    intrinsic_mat : torch.Tensor
        Camera intrinsic matrix with shape (3, 3).
    extrinsic_mat : torch.Tensor
        Camera extrinsic matrix with shape (3, 4).
    without_batch_dim : bool, optional
        If True, assumes input has no batch dimension and returns output without batch dimension.

    Returns
    -------
    torch.Tensor
        Projected points in image coordinates with shape (B, N, 3) or (N, 3) if without_batch_dim is True.
    """
    camera_points = lidar_to_camera_coordinates(
        point_cloud, intrinsic_mat, extrinsic_mat, without_batch_dim
    )
    return camera_to_image_coordinates(camera_points, without_batch_dim)


def filter_points_in_image_frame(
    points_2d: torch.Tensor,  # [B,N,3] or [N,3]
    image_width: int,
    image_height: int,
    return_mask: bool = False,
    without_batch_dim: bool = False,
) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]] | torch.Tensor:
    """
    Filter 3D points to keep only those that project within the image frame.

    Parameters
    ----------
    points_2d : torch.Tensor
        Points in image coordinates with shape (B, N, 3) or (N, 3) if without_batch_dim is True,
        where the last dimension contains (u, v, d) with u, v being pixel coordinates and d being depth.
    image_width : int
        Width of the image in pixels.
    image_height : int
        Height of the image in pixels.
    return_mask : bool, optional
        If True, also return the boolean masks used for filtering.
    without_batch_dim : bool, optional
        If True, assumes input has no batch dimension and returns output without batch dimension.

    Returns
    -------
    list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]] | torch.Tensor
        If without_batch_dim is False:
            - If return_mask is False: List of filtered points tensors
            - If return_mask is True: Tuple of (filtered points list, masks list)
        If without_batch_dim is True:
            - If return_mask is False: Filtered points tensor with shape (M, 3)
            - If return_mask is True: Tuple of (filtered points tensor, mask tensor)
    """
    # Handle case without batch dimension
    if without_batch_dim:
        if len(points_2d.shape) == 2:
            points_2d = points_2d.unsqueeze(0)
        else:
            logger.error(
                f"Invalid points_2d shape for without_batch_dim=True: {points_2d.shape}"
            )
            raise ValueError(
                f"Invalid points_2d shape for without_batch_dim=True: {points_2d.shape}"
            )

    batch_size = points_2d.shape[0]
    filtered_points = []
    masks = []

    for b in range(batch_size):
        # Extract u, v coordinates
        u = points_2d[b, :, 0]
        v = points_2d[b, :, 1]

        # Create mask for points within image boundaries
        mask = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)

        # Filter points using the mask
        valid_points = points_2d[b, mask, :]
        filtered_points.append(valid_points)
        masks.append(mask)

        if valid_points.shape[0] == 0:
            logger.warning(f"No points found in the image frame for batch {b}")

    # Return results based on flags
    if without_batch_dim:
        if return_mask:
            return filtered_points[0], masks[0]
        else:
            return filtered_points[0]
    else:
        if return_mask:
            return filtered_points, masks
        else:
            return filtered_points


def create_batched_point_cloud(
    point_cloud_list: list[torch.Tensor],
    num_points: int,
    random_seed: int = None,
    without_batch_dim: bool = False,
) -> torch.Tensor:
    """
    Create a batched point cloud tensor from a list of point clouds with varying numbers of points.
    Points are randomly sampled to ensure each batch element has the same number of points.

    Parameters
    ----------
    point_cloud_list : list[torch.Tensor] or torch.Tensor
        List of point cloud tensors, each with shape (N_i, 3) where N_i is the number of 3D points
        (which can vary between tensors). If without_batch_dim is True, this can be a single tensor
        with shape (N, 3).
    num_points : int
        Number of points to sample for each point cloud in the batch.
    random_seed : int, optional
        Random seed for reproducibility.
    without_batch_dim : bool, default=False
        If True, treats the input as a single point cloud tensor with shape (N, 3) instead of a list
        of tensors, and returns a tensor with shape (num_points, 3) instead of (1, num_points, 3).

    Returns
    -------
    torch.Tensor
        If without_batch_dim is False:
            Batched point cloud tensor with shape (B, num_points, 3) where B is the batch size.
        If without_batch_dim is True:
            Point cloud tensor with shape (num_points, 3).
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Handle case without batch dimension
    if without_batch_dim:
        if (
            isinstance(point_cloud_list, torch.Tensor)
            and len(point_cloud_list.shape) == 2
        ):
            pc = point_cloud_list
            device = pc.device

            if pc.numel() == 0:
                logger.warning("Empty point cloud provided")
                return torch.zeros((num_points, 3), device=device)

            n_points = pc.shape[0]

            if n_points <= num_points:
                logger.debug(f"Sampling with replacement ({n_points} → {num_points})")
                indices = torch.randint(0, n_points, (num_points,), device=device)
                return pc[indices]
            else:
                # If we have more points than required, sample randomly without replacement
                logger.debug(
                    f"Sampling without replacement ({n_points} → {num_points})"
                )
                indices = torch.randperm(n_points, device=device)[:num_points]
                return pc[indices]
        else:
            logger.error(
                f"Invalid input for without_batch_dim=True: {type(point_cloud_list)}"
            )
            raise ValueError(
                "Invalid input for without_batch_dim=True: expected tensor with shape (N, 3)"
            )

    batch_size = len(point_cloud_list)
    if batch_size == 0:
        raise ValueError("Empty point cloud list provided")

    # Initialize the output tensor
    device = point_cloud_list[0].device
    batched_pc = torch.zeros((batch_size, num_points, 3), device=device)

    for b, pc in enumerate(point_cloud_list):
        if pc.numel() == 0:
            logger.warning(f"Empty point cloud at batch index {b}")
            continue

        n_points = pc.shape[0]

        if n_points <= num_points:
            # If we have fewer points than required, duplicate points randomly
            logger.debug(
                f"Batch {b}: Sampling with replacement ({n_points} → {num_points})"
            )
            indices = torch.randint(0, n_points, (num_points,), device=device)
            batched_pc[b] = pc[indices]
        else:
            # If we have more points than required, sample randomly without replacement
            logger.debug(
                f"Batch {b}: Sampling without replacement ({n_points} → {num_points})"
            )
            indices = torch.randperm(n_points, device=device)[:num_points]
            batched_pc[b] = pc[indices]

    return batched_pc


def depth_image_to_point_cloud(
    depth_image: torch.Tensor,  # [1, H, W]
    rgb_image: torch.Tensor,  # [3, H, W]
    intrinsic_mat: torch.Tensor,  # [3, 3]
    extrinsic_mat: torch.Tensor = None,  # [3, 4]
) -> torch.Tensor:
    """
    Convert depth image and RGB image to a colored 3D point cloud.

    Parameters
    ----------
    depth_image : torch.Tensor
        Depth image with shape [1, H, W] where the values represent depth in meters.
    rgb_image : torch.Tensor
        RGB image with shape [3, H, W] with values in range [0, 1].
    intrinsic_mat : torch.Tensor
        Camera intrinsic matrix with shape [3, 3] containing focal lengths and principal point.
    extrinsic_mat : torch.Tensor, optional
        Camera extrinsic matrix with shape [3, 4] representing the transformation
        from world coordinates to camera coordinates. If None, points remain in camera coordinates.

    Returns
    -------
    torch.Tensor
        Colored point cloud with shape [N, 6], where N is the number of valid depth points,
        and each point contains [x, y, z, r, g, b] values.
    """
    if (
        not torch.is_tensor(depth_image)
        or depth_image.dim() != 3
        or depth_image.shape[0] != 1
    ):
        err_msg = f"depth_image must be a tensor with shape [1, H, W], got {depth_image.shape}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    if (
        not torch.is_tensor(rgb_image)
        or rgb_image.dim() != 3
        or rgb_image.shape[0] != 3
    ):
        err_msg = (
            f"rgb_image must be a tensor with shape [3, H, W], got {rgb_image.shape}"
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    device = depth_image.device

    # Extract dimensions
    _, height, width = depth_image.shape

    # Check that RGB and depth images have matching dimensions
    if rgb_image.shape[1] != height or rgb_image.shape[2] != width:
        err_msg = f"RGB image shape {rgb_image.shape} doesn't match depth image shape {depth_image.shape}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    # Create pixel coordinate grid
    v_indices, u_indices = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    u_indices = u_indices.flatten()
    v_indices = v_indices.flatten()

    # Get corresponding depth values
    z = depth_image[0, v_indices, u_indices]

    # Filter out invalid depth points (zeros or NaNs)
    valid_mask = (z > 0) & torch.isfinite(z)
    u_indices = u_indices[valid_mask]
    v_indices = v_indices[valid_mask]
    z = z[valid_mask]

    # Number of valid points
    num_points = z.shape[0]

    if num_points == 0:
        logger.warning("No valid depth points found in the image")
        return torch.zeros((0, 6), device=device)

    # Get camera parameters
    fx = intrinsic_mat[0, 0]
    fy = intrinsic_mat[1, 1]
    cx = intrinsic_mat[0, 2]
    cy = intrinsic_mat[1, 2]

    # Back-project to camera space
    x = (u_indices - cx) * z / fx
    y = (v_indices - cy) * z / fy

    # Create points in camera coordinates
    points_camera = torch.stack([x, y, z], dim=1)  # [N, 3]

    # Use points in camera coordinates if extrinsic matrix is not provided
    if extrinsic_mat is None:
        # Get RGB colors for each point
        rgb_colors = rgb_image[:, v_indices, u_indices].t()  # [N, 3]

        # Combine position and color
        colored_point_cloud = torch.cat([points_camera, rgb_colors], dim=1)  # [N, 6]

        return colored_point_cloud

    # Transform to world coordinates if extrinsic matrix is provided
    # Create homogeneous coordinates
    ones = torch.ones(num_points, 1, device=device)
    points_camera_homogeneous = torch.cat([points_camera, ones], dim=1)  # [N, 4]

    # Compute the inverse of the extrinsic matrix
    # The extrinsic matrix transforms from world to camera: P_camera = E * P_world
    # So we need the inverse to go from camera to world: P_world = E^-1 * P_camera
    R = extrinsic_mat[:3, :3]
    t = extrinsic_mat[:3, 3].unsqueeze(1)

    # Calculate inverse (R^T, -R^T * t)
    R_inv = R.transpose(0, 1)
    t_inv = -R_inv @ t

    # Construct inverse extrinsic matrix
    extrinsic_inv = torch.zeros(4, 4, device=device)
    extrinsic_inv[:3, :3] = R_inv
    extrinsic_inv[:3, 3] = t_inv.squeeze()
    extrinsic_inv[3, 3] = 1.0

    # Transform points
    points_world_homogeneous = extrinsic_inv @ points_camera_homogeneous.t()  # [4, N]
    points_world = points_world_homogeneous[:3].t()  # [N, 3]

    # Get RGB colors for each point
    rgb_colors = rgb_image[:, v_indices, u_indices].t()  # [N, 3]

    # Combine position and color
    colored_point_cloud = torch.cat([points_world, rgb_colors], dim=1)  # [N, 6]

    return colored_point_cloud
