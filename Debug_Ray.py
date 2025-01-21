import ray


tmp_dir = ray._private.utils.get_ray_temp_dir()
print(f"Ray's temporary directory: {tmp_dir}")
ray.init()
print("Ray initialization succeeded.")