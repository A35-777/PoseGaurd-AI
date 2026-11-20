"""High-performance, robust BVH parser and forward kinematics engine."""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from .skeleton import Skeleton, JointNode, AnimationClip

def _euler_matrix(rx: float, ry: float, rz: float, order: str) -> np.ndarray:
    """Compute 3x3 rotation matrix from Euler angles (in radians) according to channel order."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    
    mx = np.array([[1.0, 0.0, 0.0],
                   [0.0, cx, -sx],
                   [0.0, sx, cx]], dtype=np.float32)
                   
    my = np.array([[cy, 0.0, sy],
                   [0.0, 1.0, 0.0],
                   [-sy, 0.0, cy]], dtype=np.float32)
                   
    mz = np.array([[cz, -sz, 0.0],
                   [sz, cz, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float32)
                   
    mat = np.eye(3, dtype=np.float32)
    for axis in order:
        if axis == 'X':
            mat = mat @ mx
        elif axis == 'Y':
            mat = mat @ my
        elif axis == 'Z':
            mat = mat @ mz
    return mat

class BVHParser:
    """Parser for Biovision Hierarchy (.bvh) skeletal animation files."""
    
    @staticmethod
    def parse(file_path: str, scale_to_meters: bool = True) -> AnimationClip:
        """Parse a BVH file and compute global 3D joint positions via forward kinematics."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
        if not lines or lines[0].upper() != 'HIERARCHY':
            raise ValueError(f"Invalid BVH format in {file_path}: Expected 'HIERARCHY' header")
            
        # Parse hierarchy
        joint_names: List[str] = []
        parents: List[int] = []
        offsets: List[List[float]] = []
        nodes: Dict[str, JointNode] = {}
        
        stack: List[int] = []
        line_idx = 1
        channel_offset = 0
        end_site_counter = 0
        
        while line_idx < len(lines):
            line = lines[line_idx]
            tokens = line.split()
            first_tok = tokens[0].upper()
            
            if first_tok == 'ROOT' or first_tok == 'JOINT':
                name = tokens[1]
                parent_idx = stack[-1] if stack else -1
                current_idx = len(joint_names)
                joint_names.append(name)
                parents.append(parent_idx)
                offsets.append([0.0, 0.0, 0.0])
                
                nodes[name] = JointNode(
                    name=name,
                    parent_index=parent_idx,
                    offset=np.zeros(3, dtype=np.float32)
                )
                stack.append(current_idx)
                line_idx += 1
                
            elif first_tok == 'END' and tokens[1].upper() == 'SITE':
                parent_idx = stack[-1]
                parent_name = joint_names[parent_idx]
                end_site_counter += 1
                name = f"{parent_name}_EndSite_{end_site_counter}"
                current_idx = len(joint_names)
                joint_names.append(name)
                parents.append(parent_idx)
                offsets.append([0.0, 0.0, 0.0])
                
                nodes[name] = JointNode(
                    name=name,
                    parent_index=parent_idx,
                    offset=np.zeros(3, dtype=np.float32),
                    is_end_site=True
                )
                stack.append(current_idx)
                line_idx += 1
                
            elif first_tok == 'OFFSET':
                curr_idx = stack[-1]
                offset_vals = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
                offsets[curr_idx] = offset_vals
                curr_name = joint_names[curr_idx]
                nodes[curr_name].offset = np.array(offset_vals, dtype=np.float32)
                line_idx += 1
                
            elif first_tok == 'CHANNELS':
                curr_idx = stack[-1]
                num_channels = int(tokens[1])
                curr_channels = tokens[2:2 + num_channels]
                curr_name = joint_names[curr_idx]
                nodes[curr_name].channels = curr_channels
                nodes[curr_name].channel_indices = list(range(channel_offset, channel_offset + num_channels))
                channel_offset += num_channels
                line_idx += 1
                
            elif first_tok == '}':
                stack.pop()
                line_idx += 1
                
            elif first_tok == 'MOTION':
                line_idx += 1
                break
            else:
                line_idx += 1
                
        # Parse Motion Header
        if line_idx >= len(lines):
            raise ValueError(f"No MOTION section found in {file_path}")
            
        num_frames = 0
        frame_time = 1.0 / 30.0
        
        while line_idx < len(lines):
            line = lines[line_idx]
            if line.upper().startswith('FRAMES:'):
                num_frames = int(line.split(':')[1].strip())
                line_idx += 1
            elif line.upper().startswith('FRAME TIME:'):
                frame_time = float(line.split(':')[1].strip())
                line_idx += 1
                break
            else:
                line_idx += 1
                
        fps = 1.0 / frame_time if frame_time > 0 else 30.0
        
        # Parse Motion Frames
        motion_data_list = []
        for i in range(line_idx, len(lines)):
            row = [float(val) for val in lines[i].split()]
            if len(row) == channel_offset:
                motion_data_list.append(row)
                
        if not motion_data_list:
            raise ValueError(f"No frame data parsed in {file_path}")
            
        motion_array = np.array(motion_data_list, dtype=np.float32)
        actual_frames = motion_array.shape[0]
        
        skeleton = Skeleton(
            joint_names=joint_names,
            parents=parents,
            offsets=np.array(offsets, dtype=np.float32),
            nodes=nodes
        )
        
        # Forward Kinematics
        num_joints = len(joint_names)
        joint_positions = np.zeros((actual_frames, num_joints, 3), dtype=np.float32)
        joint_rotations = np.zeros((actual_frames, num_joints, 3), dtype=np.float32)
        root_motion = np.zeros((actual_frames, 3), dtype=np.float32)
        
        # Determine scale factor (meters vs centimeters / millimeters)
        scale = 1.0
        if scale_to_meters:
            # Check maximum offset distance in skeleton hierarchy
            max_offset = np.max(np.abs(skeleton.offsets))
            if max_offset > 15.0:  # Likely cm (typical hips height ~90-100 cm)
                scale = 0.01
            elif max_offset > 150.0: # Likely mm
                scale = 0.001
                
        skeleton.offsets = skeleton.offsets * scale
        for node in skeleton.nodes.values():
            node.offset = node.offset * scale
            
        for f in range(actual_frames):
            frame_vals = motion_array[f]
            # 4x4 global transform matrices for this frame
            global_transforms: Dict[int, np.ndarray] = {}
            
            for j_idx in range(num_joints):
                j_name = joint_names[j_idx]
                node = nodes[j_name]
                parent_idx = parents[j_idx]
                
                # Compute local translation and rotation
                local_pos = np.copy(node.offset)
                rot_order = ""
                rx, ry, rz = 0.0, 0.0, 0.0
                
                if not node.is_end_site and node.channels:
                    for ch, ch_idx in zip(node.channels, node.channel_indices):
                        val = frame_vals[ch_idx]
                        ch_upper = ch.upper()
                        if ch_upper == 'XPOSITION':
                            local_pos[0] = val * scale
                        elif ch_upper == 'YPOSITION':
                            local_pos[1] = val * scale
                        elif ch_upper == 'ZPOSITION':
                            local_pos[2] = val * scale
                        elif ch_upper == 'XROTATION':
                            rx = math.radians(val)
                            rot_order += 'X'
                        elif ch_upper == 'YROTATION':
                            ry = math.radians(val)
                            rot_order += 'Y'
                        elif ch_upper == 'ZROTATION':
                            rz = math.radians(val)
                            rot_order += 'Z'
                            
                rot_mat = _euler_matrix(rx, ry, rz, rot_order)
                joint_rotations[f, j_idx] = [rx, ry, rz]
                
                local_tf = np.eye(4, dtype=np.float32)
                local_tf[:3, :3] = rot_mat
                local_tf[:3, 3] = local_pos
                
                if parent_idx == -1:
                    global_tf = local_tf
                    root_motion[f] = local_pos
                else:
                    parent_tf = global_transforms[parent_idx]
                    global_tf = parent_tf @ local_tf
                    
                global_transforms[j_idx] = global_tf
                joint_positions[f, j_idx] = global_tf[:3, 3]
                
        return AnimationClip(
            skeleton=skeleton,
            fps=fps,
            frame_time=frame_time,
            num_frames=actual_frames,
            joint_positions=joint_positions,
            joint_rotations=joint_rotations,
            root_motion=root_motion,
            metadata={
                "source_file": file_path,
                "scale_applied": scale,
                "channel_count": channel_offset
            }
        )

    @staticmethod
    def export(clip: AnimationClip, output_path: str) -> None:
        """Export an AnimationClip to standard BVH format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("HIERARCHY\n")
            skeleton = clip.skeleton
            
            def write_joint(j_idx: int, indent: str):
                name = skeleton.joint_names[j_idx]
                node = skeleton.nodes.get(name)
                parent_idx = skeleton.parents[j_idx]
                
                if node and node.is_end_site:
                    f.write(f"{indent}End Site\n{indent}{{\n")
                    f.write(f"{indent}\tOFFSET {node.offset[0]:.6f} {node.offset[1]:.6f} {node.offset[2]:.6f}\n")
                    f.write(f"{indent}}}\n")
                    return
                    
                kw = "ROOT" if parent_idx == -1 else "JOINT"
                f.write(f"{indent}{kw} {name}\n{indent}{{\n")
                off = skeleton.offsets[j_idx]
                f.write(f"{indent}\tOFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}\n")
                
                if parent_idx == -1:
                    f.write(f"{indent}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
                else:
                    f.write(f"{indent}\tCHANNELS 3 Zrotation Xrotation Yrotation\n")
                    
                children = [idx for idx, p in enumerate(skeleton.parents) if p == j_idx]
                for child_idx in children:
                    write_joint(child_idx, indent + "\t")
                    
                f.write(f"{indent}}}\n")
                
            write_joint(0, "")
            
            f.write("MOTION\n")
            f.write(f"Frames: {clip.num_frames}\n")
            f.write(f"Frame Time: {clip.frame_time:.6f}\n")
            
            # Simple motion exporter
            for t in range(clip.num_frames):
                row = []
                # Root position and rotation
                root_pos = clip.root_motion[t] if clip.root_motion is not None else clip.joint_positions[t, 0]
                row.extend([root_pos[0], root_pos[1], root_pos[2]])
                
                for j_idx in range(skeleton.num_joints):
                    node = skeleton.nodes.get(skeleton.joint_names[j_idx])
                    if node and node.is_end_site:
                        continue
                    if clip.joint_rotations is not None:
                        rot = clip.joint_rotations[t, j_idx]
                        deg_x, deg_y, deg_z = math.degrees(rot[0]), math.degrees(rot[1]), math.degrees(rot[2])
                        row.extend([deg_z, deg_x, deg_y])
                    else:
                        row.extend([0.0, 0.0, 0.0])
                f.write(" ".join(f"{val:.6f}" for val in row) + "\n")

