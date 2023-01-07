import numpy as np
import open3d as o3d
import copy

from icp import icp,draw_registration_result
from nricp import nonrigidIcp



#read source file
data_num = 1
chest_pcd_ct = o3d.io.read_point_cloud('/home/xuesong/CAMP/pointcloud_data/CT_pointcloud_som/'+str(data_num)+'/surface.ply')
ct_rib = o3d.geometry.PointCloud()
ct_rib.points = o3d.utility.Vector3dVector(np.array(chest_pcd_ct.points)) # CT原始点云
# o3d.geometry.estimate_normals(ct_rib, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# ct_rib.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
ct_rib.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# ct_rib.orient_normals_consistent_tangent_plane(100)

chest_pcd_us = o3d.io.read_point_cloud('/home/xuesong/CAMP/pointcloud_data/tissue_filter_simplification(near).ply')
us_rib = o3d.geometry.PointCloud()
us_rib.points = o3d.utility.Vector3dVector(np.array(chest_pcd_us.points)) # US原始点云
# o3d.geometry.estimate_normals(us_rib, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# us_rib.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
us_rib.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# us_rib.orient_normals_consistent_tangent_plane(100)

alpha = 4
mesh_ct = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(ct_rib, 4)
mesh_us = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(us_rib, alpha)

o3d.visualization.draw_geometries([mesh_ct, mesh_us], window_name = '原始mesh_ct, mesh_us')  
# sourcemesh = o3d.io.read_triangle_mesh("data/source_test.obj")
# targetmesh = o3d.io.read_triangle_mesh("data/target_half.obj")
sourcemesh = mesh_ct
targetmesh = mesh_us
sourcemesh.compute_vertex_normals()
targetmesh.compute_vertex_normals()



#first find rigid registration
# guess for inital transform for icp
initial_guess = np.eye(4)
affine_transform = icp(sourcemesh,targetmesh,initial_guess)


#creating a new mesh for non rigid transform estimation 
refined_sourcemesh = copy.deepcopy(sourcemesh)
refined_sourcemesh.transform(affine_transform)
refined_sourcemesh.compute_vertex_normals()


#non rigid registration
deformed_mesh = nonrigidIcp(refined_sourcemesh,targetmesh)



sourcemesh.paint_uniform_color([0.1, 0.9, 0.1]) # green
targetmesh.paint_uniform_color([0.9,0.1,0.1]) # red
deformed_mesh.paint_uniform_color([0.1,0.1,0.9]) # blue
# o3d.visualization.draw_geometries([targetmesh,deformed_mesh]) # 
o3d.visualization.draw_geometries([targetmesh, deformed_mesh]) # 


