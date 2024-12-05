from torch.distributed.device_mesh import init_device_mesh
mesh_2d = init_device_mesh("cuda", mesh_shape=(2, 2), mesh_dim_names=("ep", "tp"))

mesh_2d.get_group(mesh_dim="ep")
mesh_2d.get_group(mesh_dim="tp")
