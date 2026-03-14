from utils.general_utils import *
import os
import torch
import numpy
import open3d as o3d
import trimesh
from tqdm import tqdm

def get_aligned_models(scene_anno, loader='o3d', visualize=False, scene_pcd=None,  shapenet_path="data/objects/ShapeNetV2"):
    aligned_models = []
    for model in scene_anno['aligned_models']:
        cad_path = os.path.join(shapenet_path, model['catid_cad'], model['id_cad'], "models", "model_normalized.obj")
        if loader=='o3d':
            cad = o3d.io.read_triangle_mesh(cad_path)
        else:
            cad = trimesh.load_mesh(cad_path, force='mesh')
            if isinstance(cad, trimesh.Scene):
                mesh_list = []
                for g in cad.geometry:
                    mesh_list.append(cad.geometry[g])
                cad = trimesh.util.concatenate(mesh_list)
            cad = cad.as_open3d
        # print(f"\tLoaded CAD models: o3d: {cad.get_center()}, trimesh centroid: {cad_trimesh.centroid}, com: {cad_trimesh.center_mass}")
        cad_trans_mat = calc_Mcad(model)
        cad = cad.transform(cad_trans_mat)
        aligned_models.append({'cad':cad, 'catid_cad': model['catid_cad'], 'id_cad': model['id_cad']})
    if visualize:
        if scene_pcd is None:
            scene_pcd = []
        else:
            scene_pcd = [scene_pcd]
        if loader!='o3d':
            o3d_models = []
            for a in aligned_models:
                mesh = a['cad']
                o3d_models.append(PCFromArray(np.array(mesh.vertices), [0,0,1]))                
            o3d.visualization.draw_geometries(scene_pcd+o3d_models)
        else:
            o3d.visualization.draw_geometries(scene_pcd+aligned_models)
    return aligned_models

def decompose_mat4(M):
    R = M[0:3, 0:3]
    sx = np.linalg.norm(R[0:3, 0])
    sy = np.linalg.norm(R[0:3, 1])
    sz = np.linalg.norm(R[0:3, 2])

    s = np.array([sx, sy, sz])

    R[:, 0] /= sx;
    R[:, 1] /= sy;
    R[:, 2] /= sz;
    rot_mat = R[0:3, 0:3]
    q = quaternion.from_rotation_matrix(rot_mat)

    t = M[0:3, 3]
    return t, [q.w, q.x, q.y, q.z], s

def get_aligned_bbox(cad_anno):
    bbox_mat = calc_Mbbox(cad_anno)
    t, quat, s = decompose_mat4(bbox_mat)
    obb = o3d.geometry.OrientedBoundingBox(t, o3d.geometry.get_rotation_matrix_from_quaternion(quat), s*2)
    obb.color = [1,0,0]
    return obb
                        
def shapenet2scannet_name(s2c_names):
    valid_s2c_name = None
    for n in s2c_names:
        if n in SCANNETV2_NAMES:
            valid_s2c_name = n
            return valid_s2c_name        
    return None

def obb_based_overlap(obb, pcd):
    inliers = obb.get_point_indices_within_bounding_box(pcd.points)
    return len(inliers)/len(pcd.points)

def calc_instance_center_bbox(model):
    Mbbox = calc_Mbbox(model)
    t, r_mat, s = decompose_mat4(Mbbox)
    return np.array(t)

if __name__=='__main__':
    annotations_path = "scan2cad_annotator/scan2cad_v2_annotations.json"
    scannet_path = "data/objects/ScanNet_v2/dataset"
    shapenet_path = "data/objects/ShapeNetV2"
    v2_annotations_folder = 'scan2cad_annotator/per_scene'

    overlap_max_thresh = .7
    overlap_min_thresh = 0.05

    # Load scan2cad annotations
    assert annotations_path is not None and os.path.exists(annotations_path), "S2C Annotations path is None or does not exist!"
    scan2cad_annos = LoadJson(annotations_path)

    # Load ShapeNet taxonomy
    shapenet_id2names, shapenet_name2id = LoadShapeNetTaxonomy(os.path.join(shapenet_path, "taxonomy.json"))

    # Find all annotated scenes
    annotated_files = GetFilesInDir(v2_annotations_folder, prefix='s2c_v2_', suffix='.json')
    annotated_scenes = [os.path.basename(f)[len("s2c_v2_"):-len('.json')] for f in annotated_files]
    annotated_scenes = set(annotated_scenes)
    # annotated_scenes = {}

    # Get all annotated ScanNet data
    phases = ['train', 'val']
    scannet_files = {}
    for p in phases:
        root_dir = os.path.join(scannet_path, p)
        suffix = "_inst_nostuff_v2.pth"
        file_list = GetFilesInDir(root_dir, suffix)
        scan_ids = set([os.path.basename(f)[:-len(suffix)] for f in file_list])
        scannet_files[p] = scan_ids
        print(f"Found {len(scannet_files[p])} files for phase {p}")

    auto_anno_cnt = 0
    auto_disc_cnt = 0
    manual_anno_cnt = 0
    unmatched_categories = {}
    max_reject_overlap = 0

    for scene_id in tqdm(scan2cad_annos.keys()):
        # skip scene if scannet instance annotation exists
        if not scene_id in annotated_scenes:
            scene_s2c_anno = scan2cad_annos[scene_id]
            
            # check which phase the scan belongs to
            phase = None
            for p in phases:
                if scene_id in scannet_files[p]:
                    phase = p
                    break
            
            # Skip file if couldn't find phase
            if phase is None:
                print(f"Couldn't find phase for scan: {scene_id}")
                continue
                    
            # Load scannet annotations
            scannet_scene_anno_path = os.path.join(scannet_path, phase, scene_id+suffix)
            assert os.path.exists(scannet_scene_anno_path), f"File path doesn't exist: {scannet_scene_anno_path}" 
            sgt_coords, sgt_coords_raw, sgt_colors, sgt_sem_labels, sgt_inst_labels = torch.load(scannet_scene_anno_path)
            sgt_sem_labels = np.array(sgt_sem_labels, dtype=int)
            sgt_inst_labels = np.array(sgt_inst_labels, dtype=int)
            
            # Load and Transform scan using S2C scene TRS
            scene_pcd = PCFromArray(sgt_coords_raw, sgt_colors)
            Mscan = calc_Mcad(scene_s2c_anno)
            scene_pcd = scene_pcd.transform(Mscan)                     
            scene_points = np.array(scene_pcd.points)
            scene_colors = np.array(scene_pcd.colors)
            
            # Get aligned CAD models from Scan2CAD annotations
            aligned_cad_models = get_aligned_models(scene_s2c_anno, scene_pcd=scene_pcd, loader='trimesh', visualize=False, shapenet_path=shapenet_path)
            aligned_model_centers = np.array([m['cad'].get_center() for m in aligned_cad_models])
            aligned_bboxes = [get_aligned_bbox(a) for a in scene_s2c_anno['aligned_models']]
            
            cads = [m['cad'] for m in aligned_cad_models]
            # aligned_bboxes = [get_aligned_bbox(a) for a in scene_s2c_anno['aligned_models']]
            # ShowPointCloud(cads+aligned_bboxes)
            
            # Extract CAD model vertices and category names  
            cad_categories = []
            for m in aligned_cad_models:
                m['vertices'] = np.array(m['cad'].vertices)
                s2c_names = shapenet_id2names[m['catid_cad']]['name']
                valid_name = shapenet2scannet_name(s2c_names=s2c_names)
                if valid_name is not None:
                    cad_categories.append(valid_name)
                else:
                    # print(f"Unable to find valid name for: {s2c_names}")
                    if m['catid_cad'] not in unmatched_categories.keys():
                        unmatched_categories[m['catid_cad']] = s2c_names
                        
            # print()
            cad_categories = set(cad_categories)
            
            # Iterate through unique instances in scannet and extract instances with the same category as CAD models
            unique_inst_labels = np.unique(sgt_inst_labels)
            scannet_instances = []
            num_points = 2048
            for i in unique_inst_labels:
                if i<0:
                    continue
                mask = sgt_inst_labels == i
                inst_sem_label = GetDominantLabel(sgt_sem_labels[mask])
                inst_sem_name = SCANNETV2_LABEL2NAME[inst_sem_label]            
                                
                inst_points = scene_points[mask]
                # inst_norm_points = scene_norm_points[mask]
                inst_colors = scene_colors[mask]
                # ShowPointCloud(PCFromArray(inst_points, inst_colors), window_name=f"Inst #{i}: {inst_sem_name}")
                
                resampled_points = inst_points.copy()
                resampled_colors = inst_colors.copy()
                # # resampled_inst_norm_points = inst_norm_points.copy()
                # if inst_points.shape[0] < num_points:
                #     resampled_points = np.array(inst_points.tolist()*((num_points//inst_points.shape[0])+1))
                #     resampled_colors = np.array(inst_colors.tolist()*((num_points//inst_colors.shape[0])+1))
                #     # inst_norm_points = np.array(inst_norm_points.tolist()*((num_points//inst_norm_points.shape[0])+1))

                # rand_inds = random.sample([i for i in range(resampled_points.shape[0])], num_points)
                # resampled_points = resampled_points[rand_inds, :]
                # resampled_colors = resampled_colors[rand_inds, :]
                # # resampled_inst_norm_points = resampled_inst_norm_points[rand_inds, :]

                scannet_instances.append({'points':resampled_points, 'colors':resampled_colors, 'pcd':PCFromArray(resampled_points, resampled_colors), 'center':resampled_points.mean(0), 'sem_name':inst_sem_name, 'inst_id':i})
            
            scannet_centers = np.array([m['center'] for m in scannet_instances])
            scannet_inst_ids = [m['inst_id'] for m in scannet_instances]
            scannet_inst_pcds = [m['pcd'] for m in scannet_instances]
            
            rand_colors = GenerateRandomColors(len(scannet_inst_pcds))
            colored_scannet_insts = [PCFromArray(np.array(pcd.points), np.array(rand_colors[idx])) for idx, pcd in enumerate(scannet_inst_pcds)]
            
            # # Visualize instances and CAD models together
            # cad_meshes = [m['cad'] for m in aligned_cad_models]
            # obj_pcds = [PCFromArray(inst['points'], inst['colors']) for inst in scannet_instances]
            # ShowPointCloud(cad_meshes+obj_pcds)

            # Iterate through aligned CAD models
            matched_pcds = []
            n_models = len(aligned_cad_models)
            for idx, model in enumerate(aligned_cad_models):
                # obb = aligned_bboxes[idx]
                # overlap_scores = [obb_based_overlap(obb, inst) for inst in scannet_inst_pcds]
                # max_overlap_idx = np.argmax(overlap_scores)
                # max_overlap = overlap_scores[max_overlap_idx]
                # window_name = f"Scene: {scene_id}, obj #{idx}/{n_models}, Overlap: {max_overlap}"
                # ShowPointCloud([obb, model['cad'], scannet_inst_pcds[max_overlap_idx]], window_name=window_name, axes_size=1)
                
                # model_center = model['cad'].get_center()
                model_center = calc_instance_center_bbox(scene_s2c_anno['aligned_models'][idx])
                distances = np.sqrt(((scannet_centers - model_center)**2).sum(axis=1))
                # dist_list = sorted([[d,i] for i,d in enumerate(distances)])
                dist_list = sorted(list(zip(distances, scannet_inst_ids, scannet_inst_pcds)))
                
                k = 5
                # valid_insts = dist_list.copy()            
                # # Select closest k instances to the CAD model
                # if len(valid_insts)>k:
                #     valid_insts = valid_insts[:k]
                valid_insts = dist_list[:k]
                
                # For each selected instance, compute voxel overlap
                cad_pcd = PCFromArray(model['vertices'])
                inst_overlaps = []
                max_overlap=0
                overlap_inst = None
                for inst in valid_insts:
                    instance = scannet_instances[inst[1]]
                    overlap = ComputePointcloudOverlap(cad_pcd, instance['pcd'])
                    
                    # ShowPointCloud([model['cad'], instance['pcd']], window_name=f"Overlap: {overlap}")
                    inst_overlaps.append([overlap, inst])
                    if overlap>max_overlap:
                        max_overlap = overlap
                        overlap_inst = inst
                
                if max_overlap<=overlap_min_thresh:
                    auto_disc_cnt+=1
                    scene_s2c_anno['aligned_models'][idx]['scannet_instance_id'] = -1
                    max_reject_overlap = max(max_reject_overlap, max_overlap)
                elif max_overlap>overlap_max_thresh:
                    auto_anno_cnt+=1
                    scene_s2c_anno['aligned_models'][idx]['scannet_instance_id'] = int(overlap_inst[1])
                    matched_pcds.extend([model['cad'], overlap_inst[2]])
                    # ShowPointCloud([model['cad'], overlap_inst[2]])
                elif scene_s2c_anno['aligned_models'][idx]['scannet_instance_id'] >= 0:
                # else:
                    manual_anno_cnt+=1
                    # scene_s2c_anno['aligned_models'][idx]['scannet_instance_id'] = -1
                    
                    for io in inst_overlaps:
                        overlap_inst = io[1]
                        max_overlap = io[0]
                        if max_overlap>=overlap_min_thresh:
                            window_name = f"Scene: {scene_id}, obj #{idx}/{n_models}, Overlap: {max_overlap}, #points: {len(overlap_inst[2].points)}"
                            # Show visualization for manual vetting
                            # ShowPointCloud([model['cad'], overlap_inst[2]], window_name=f"Overlap: {max_overlap}")
                            # scene_s2c_anno['aligned_models'][idx]['scannet_instance_id'] = int(overlap_inst[1])
                            scene_s2c_anno['aligned_models'][idx]['scannet_instance_id'] = -1
                            
                            # Draw the point cloud and wait for the window to be closed
                            visualizer = o3d.visualization.VisualizerWithKeyCallback()
                            visualizer.create_window(window_name=window_name, width=1920, height=1080, left=800, top=800)
                            visualizer.add_geometry(model['cad'])
                            visualizer.add_geometry(overlap_inst[2])
                            
                            # Setup callbacks
                            def accept_callback(vis):
                                # This function will be called on a key press
                                print("Accept Key Pressed!")
                                scene_s2c_anno['aligned_models'][idx]['scannet_instance_id'] = int(overlap_inst[1])
                                # matched_pcds.extend([model['cad'], overlap_inst[2]])
                                # visualizer.destroy_window()
                                visualizer.close()
                                
                            def reject_callback(vis):
                                # This function will be called on a key press
                                # if max_reject_overlap<max_overlap:
                                #     max_reject_overlap = max_overlap
                                # max_reject_overlap = max(max_reject_overlap, max_overlap)
                                
                                print(f"Reject: {max_overlap} max overlap so far {max_reject_overlap}")
                                scene_s2c_anno['aligned_models'][idx]['scannet_instance_id'] = -1
                                # visualizer.destroy_window()
                                visualizer.close()
                                
                            visualizer.register_key_callback(ord('W'), accept_callback)
                            visualizer.register_key_callback(ord('R'), reject_callback)                            
                            #Run visualizer
                            # visualizer.set_full_screen(True)
                            visualizer.run()
                            visualizer.destroy_window()
                            
                            if scene_s2c_anno['aligned_models'][idx]['scannet_instance_id']!=-1:
                                break
                            else:
                                max_reject_overlap = max(max_reject_overlap, max_overlap)
            
            # ShowPointCloud(matched_pcds)
            out_path = os.path.join(v2_annotations_folder, "s2c_v2_"+scene_id+".json")
            WriteJson(scene_s2c_anno, out_path)
                    
    print(f"Total objects: {auto_anno_cnt+auto_disc_cnt+manual_anno_cnt}, Auto anno:{auto_anno_cnt}, disc: {auto_disc_cnt}, manual: {manual_anno_cnt}")
    print()        

        
    """
    TODO:
    [ ] If overlap > max_thresh -> Auto annotate
    [ ] If overlap < min_thresh -> discard
    [ ] If overlap between max and min thresh -> Manual
    """