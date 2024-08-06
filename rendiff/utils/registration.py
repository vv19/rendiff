import torch
import numpy as np


def get_rigid_transforms_torch(pcd_a, pcd_b):
    '''
    pcd_a: (B, N, 3)
    pcd_b: (B, N, 3)
    Compute the rigid transformation from pcd_a to pcd_b using SVD (aruns method).
    Returns:
    T: (B, 4, 4)
    '''
    p1, p2 = pcd_a, pcd_b
    p1_centroid = p1.mean(dim=1, keepdim=True)
    p2_centroid = p2.mean(dim=1, keepdim=True)
    p1_prime = p1 - p1_centroid
    p2_prime = p2 - p2_centroid
    H = torch.bmm(p1_prime.transpose(1, 2), p2_prime)
    U, S, V = torch.svd(H)

    D = torch.eye(3, device=pcd_a.device).unsqueeze(0).repeat(pcd_a.shape[0], 1, 1)
    D[:, 2, 2] = torch.det(V @ U.transpose(1, 2))
    R = V @ D @ U.transpose(1, 2)

    t = p2_centroid - torch.bmm(R, p1_centroid.transpose(1, 2)).transpose(1, 2)
    T = torch.eye(4, device=pcd_a.device).unsqueeze(0).repeat(pcd_a.shape[0], 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t.squeeze(-2)
    return T


def get_rigid_transform(pcd_a, pcd_b):
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        return get_rigid_transforms_torch(pcd_a[None, :, :], pcd_b[None, :, :]).squeeze(0)


def get_rigid_transforms_np(pcd_a, pcd_b):
    p1, p2 = pcd_a.T.detach().cpu().numpy(), pcd_b.T.detach().cpu().numpy()

    R, t = arun(p1, p2)

    T = torch.eye(4, device=pcd_a.device)
    T[:3, :3] = torch.tensor(R, device=pcd_a.device).float()
    T[:3, 3] = torch.tensor(t, device=pcd_a.device).float().squeeze()
    return T


# Really needs to be rewritten in PyTorch...
def arun(A, B):
    """Solve 3D registration using Arun's method: B = RA + t
    """
    N = A.shape[1]
    assert B.shape[1] == N

    # calculate centroids
    A_centroid = np.reshape(1 / N * (np.sum(A, axis=1)), (3, 1))
    B_centroid = np.reshape(1 / N * (np.sum(B, axis=1)), (3, 1))

    # calculate the vectors from centroids
    A_prime = A - A_centroid
    B_prime = B - B_centroid

    # rotation estimation
    H = np.zeros([3, 3])
    for i in range(N):
        ai = A_prime[:, i]
        bi = B_prime[:, i]
        H = H + np.outer(ai, bi)
    U, S, V_transpose = np.linalg.svd(H)
    V = np.transpose(V_transpose)
    U_transpose = np.transpose(U)
    R = V @ np.diag([1, 1, np.linalg.det(V) * np.linalg.det(U_transpose)]) @ U_transpose

    # translation estimation
    t = B_centroid - R @ A_centroid

    return R, t
