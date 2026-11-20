"""Procedurally build a werewolf from an MPFB human base and export it as GLB.
Usage: blender --background --python build_werewolf.py
Requires the MPFB extension (blender --online-mode --command extension install --enable mpfb).
"""

import math
import os
import random

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector, noise

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_TRIANGLES = 30000
FUR_CARD_COUNT = 20000
VOXEL_SIZE = 0.008
TUFT_COUNT = 340
HEAD_SCALE = 1.2
FUR_THICKNESS = 0.02
Z_AXIS = Vector((0, 0, 1))
DANCE_FRAMES = 96  # eight beats at 120 bpm and 24 fps, loops seamlessly
STILL_FRAME = 18
TAIL_BONES = 4
JAW_HINGE, JAW_TIP = (0, 0.06, 0.035), (0, 0.292, -0.037)  # lower jaw centre line in the head frame
EYE, NOSE = (0.052, 0.142, 0.137), (0, 0.335, 0.082)  # left eye (the right one mirrors it) and nose pad in the head frame
# Teeth of one side as (x, y, length, radius, thickness across the jaw as a share of the radius), from the front of the jaw back:
# three incisors, the canine, premolars, the two-cusped carnassial and a molar
UPPER_TEETH = [(0.004, 0.328, 0.010, 0.0045, 0.8), (0.011, 0.326, 0.010, 0.0045, 0.8), (0.018, 0.320, 0.013, 0.005, 0.8),
               (0.028, 0.288, 0.046, 0.0095, 0.85),
               (0.031, 0.262, 0.011, 0.006, 0.6), (0.034, 0.243, 0.013, 0.007, 0.6), (0.037, 0.222, 0.015, 0.008, 0.6),
               (0.040, 0.196, 0.020, 0.012, 0.5), (0.041, 0.181, 0.012, 0.008, 0.6), (0.041, 0.165, 0.009, 0.008, 0.8)]
LOWER_TEETH = [(0.0035, 0.3125, 0.009, 0.004, 0.8), (0.0095, 0.3105, 0.009, 0.004, 0.8), (0.015, 0.306, 0.011, 0.0045, 0.8),
               (0.020, 0.299, 0.040, 0.0085, 0.85),
               (0.0235, 0.277, 0.009, 0.005, 0.6), (0.0265, 0.258, 0.012, 0.006, 0.6), (0.029, 0.238, 0.014, 0.007, 0.6),
               (0.031, 0.217, 0.015, 0.0075, 0.6),
               (0.0335, 0.191, 0.019, 0.011, 0.5), (0.034, 0.177, 0.012, 0.008, 0.6), (0.034, 0.162, 0.008, 0.007, 0.8)]
SKIN_BLEND = 0.025  # metres over which neighbouring bones share a vertex
# Approximate body radius around each skinning bone (side suffix and tail number removed)
BONE_RADIUS = {"pelvis": 0.13, "spine_01": 0.14, "spine_02": 0.15, "spine_03": 0.17, "neck_01": 0.09, "head": 0.10,
               "clavicle": 0.06, "upperarm": 0.06, "lowerarm": 0.05, "hand": 0.04, "thigh": 0.09, "calf": 0.06,
               "foot": 0.04, "ball": 0.03, "tail": 0.08}
random.seed(11)

# Hunched, digitigrade stance as (bone, armature-space axis, degrees), parents before children
POSE = [
    ("spine_01", "X", 15), ("spine_02", "X", 12), ("spine_03", "X", 10), ("neck_01", "X", -12), ("head", "X", -22),
    ("upperarm_l", "Y", 15), ("upperarm_l", "X", -50), ("lowerarm_l", "X", -60),
    ("upperarm_r", "Y", -15), ("upperarm_r", "X", -50), ("lowerarm_r", "X", -60),
    ("thigh_l", "Y", -10), ("thigh_l", "X", -50), ("calf_l", "X", 75), ("foot_l", "X", 5), ("ball_l", "X", -30),
    ("thigh_r", "Y", 10), ("thigh_r", "X", -50), ("calf_r", "X", 75), ("foot_r", "X", 5), ("ball_r", "X", -30),
]
# Uniform bone scales: barrel chest, shrunken human head (hidden inside the wolf skull), big paws
BONE_SCALES = {"spine_03": 1.12, "head": 0.65, "hand_l": 1.3, "hand_r": 1.3, "foot_l": 1.25, "foot_r": 1.25}
# How likely each body region is to grow a fur tuft
TUFT_WEIGHTS = {"spine_03": 1.0, "spine_02": 0.7, "spine_01": 0.5, "neck_01": 1.0, "clavicle": 1.0, "upperarm": 0.8,
                "lowerarm": 0.9, "thigh": 0.6, "calf": 0.7, "pelvis": 0.5}


def make_material(name, color, roughness=0.5, emission=0.0, vertex_color=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    if emission:
        bsdf.inputs["Emission Color"].default_value = (*color, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission
    if vertex_color:
        attr = mat.node_tree.nodes.new("ShaderNodeVertexColor")
        attr.layer_name = vertex_color
        mat.node_tree.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def add_sphere(bm, frame, center, radii, detail=1):
    """Add an ellipsoid whose centre and radii are given in the coordinates of `frame`."""
    matrix = frame @ Matrix.Translation(center) @ Matrix.Diagonal((*radii, 1.0))
    bmesh.ops.create_uvsphere(bm, u_segments=20 * detail, v_segments=12 * detail, radius=1.0, matrix=matrix)


def chain(p0, p1, r0, r1, count, flatten=(1.0, 1.0, 1.0)):
    """Return (centre, radii) for each ellipsoid of a tapered, overlapping row from p0 to p1."""
    p0, p1 = Vector(p0), Vector(p1)
    spheres = []
    for i in range(count):
        t = i / (count - 1)
        r = r0 + (r1 - r0) * t
        spheres.append((p0.lerp(p1, t), Vector((r * flatten[0], r * flatten[1], r * flatten[2]))))
    return spheres


def add_chain(bm, frame, *row, **options):
    """Add a tapered row of overlapping ellipsoids from p0 to p1."""
    for centre, radii in chain(*row, **options):
        add_sphere(bm, frame, centre, radii)


# Ellipsoids of the two halves of the muzzle in the head frame, shared by the fused mesh and the teeth planted on it;
# the upper half has a lip pad down each side that hangs over the roots of the teeth
UPPER_JAW = chain((0, 0.13, 0.088), (0, 0.31, 0.078), 0.058, 0.04, 8, flatten=(1.0, 1.0, 0.8))
UPPER_JAW += [s for sx in (-1, 1) for s in chain((sx * 0.03, 0.17, 0.07), (sx * 0.02, 0.31, 0.068), 0.03, 0.022, 6)]
LOWER_JAW = chain(JAW_HINGE, JAW_TIP, 0.055, 0.028, 9, flatten=(1.0, 1.0, 0.6))


def jaw_surface(spheres, x, y, side):
    """Height in the head frame of the top (side 1) or underside (side -1) of a row of ellipsoids above the point (x, y)."""
    heights = []
    for centre, radii in spheres:
        inside = 1 - ((x - centre.x) / radii.x) ** 2 - ((y - centre.y) / radii.y) ** 2
        if inside > 0:
            heights.append(centre.z + side * radii.z * math.sqrt(inside))
    return side * max(side * h for h in heights)


def smoothstep(x, low, high):
    t = min(1.0, max(0.0, (x - low) / (high - low)))
    return t * t * (3 - 2 * t)


def parting(y):
    """Height in the head frame of the line where the jaws part, at a distance y along the muzzle."""
    return 0.036 - 0.16 * (y - 0.13)


def mouth_mask(local, normal_z):
    """Return (how much a point of the fused head is lip, how much is the inside of the mouth), each from 0 to 1."""
    gap = parting(local.y)
    if not (0.11 < local.y < 0.37 and abs(local.x) < 0.07 and gap - 0.07 < local.z < gap + 0.055):
        return 0.0, 0.0
    facing = normal_z if local.z < gap else -normal_z  # how squarely the surface faces the other jaw
    fade = smoothstep(local.y, 0.11, 0.15)
    return fade * smoothstep(facing, 0.1, 0.45), fade * smoothstep(facing, 0.5, 0.8)


def add_loft(bm, points, radii, across=(1, 0, 0), flat=1.0, segments=8):
    """Add a tube of rings around a line of points; a radius of zero closes the tube to a point there.
    The rings are `flat` times as thick at right angles to `across` as they are along it."""
    points, across = [Vector(p) for p in points], Vector(across)
    rings = []
    for i, (point, radius) in enumerate(zip(points, radii)):
        if not radius:
            rings.append([bm.verts.new(point)])
            continue
        axis = (points[min(i + 1, len(points) - 1)] - points[max(i - 1, 0)]).normalized()
        u = (across - axis * across.dot(axis)).normalized()
        v = axis.cross(u)
        angles = [2 * math.pi * k / segments for k in range(segments)]
        rings.append([bm.verts.new(point + (u * math.cos(a) + v * math.sin(a) * flat) * radius) for a in angles])
    for a, b in zip(rings, rings[1:]):
        for k in range(segments):
            j = (k + 1) % segments
            # A ring of one vertex repeats that vertex, which leaves a triangle
            bm.faces.new(list(dict.fromkeys((a[k % len(a)], a[j % len(a)], b[j % len(b)], b[k % len(b)]))))


def add_cone(bm, base, tip, radius, segments=6):
    """Add a closed cone from `base` to a point at `tip` (world coordinates)."""
    base, tip = Vector(base), Vector(tip)
    axis = tip - base
    matrix = Matrix.Translation((base + tip) / 2) @ axis.to_track_quat("Z", "Y").to_matrix().to_4x4()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segments, radius1=radius, radius2=0.0, depth=axis.length, matrix=matrix)


def bmesh_to_object(bm, name):
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def join_objects(objs, name):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    joined = bpy.context.active_object
    joined.name = name
    return joined


def apply_modifier(obj, modifier):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def posed_human():
    """Create a muscular MPFB human, pose it, and return (static body object, armature)."""
    from bl_ext.blender_org.mpfb.services.humanservice import HumanService
    from bl_ext.blender_org.mpfb.services.targetservice import TargetService

    macro = TargetService.get_default_macro_info_dict()
    macro.update(gender=1.0, muscle=1.0, weight=0.6, height=0.8)
    body = HumanService.create_human(macro_detail_dict=macro)
    rig = HumanService.add_builtin_rig(body, "game_engine")

    for bone, axis, degrees in POSE:
        pb = rig.pose.bones[bone]
        pivot = Matrix.Translation(pb.matrix.translation)
        pb.matrix = pivot @ Matrix.Rotation(math.radians(degrees), 4, axis) @ pivot.inverted() @ pb.matrix
        bpy.context.view_layer.update()
    for bone, scale in BONE_SCALES.items():
        rig.pose.bones[bone].scale = (scale, scale, scale)
    bpy.context.view_layer.update()

    # Bake shape keys, helper mask and pose into a plain mesh
    depsgraph = bpy.context.evaluated_depsgraph_get()
    baked = bpy.data.meshes.new_from_object(body.evaluated_get(depsgraph), preserve_all_data_layers=True, depsgraph=depsgraph)
    # Inflate the body along its normals to stand in for the thickness of the coat
    for v in baked.vertices:
        v.co += v.normal * FUR_THICKNESS
    body.shape_key_clear()
    body.data = baked
    body.modifiers.clear()
    body.parent = None
    body.matrix_world = Matrix.Identity(4)
    return body, rig


def head_frame(rig):
    """Frame at the skull base: x across the head, y out through the muzzle, z up through the crown."""
    pb = rig.pose.bones["head"]
    up = (pb.tail - pb.head).normalized()
    forward = Vector((0, -1, 0))
    forward = (forward - up * forward.dot(up)).normalized()
    right = forward.cross(up)
    return Matrix.Translation(pb.head) @ Matrix((right, forward, up)).transposed().to_4x4() @ Matrix.Scale(HEAD_SCALE, 4)


def tail_point(root, t):
    """Point on the centre line of the drooping tail, t from 0 at the root to 1 at the tip."""
    return root + Vector((0, 0.66 * t, 0.12 * math.sin(t * math.pi) - 0.22 * t * t))


def build_wolf_parts(body, rig, head, tail_root):
    """Return a bmesh of everything fused with the body: wolf head, neck, hump, tail and fur tufts."""
    bm = bmesh.new()
    world = Matrix.Identity(4)
    bone = rig.pose.bones

    # Low skull with a sloping forehead, muzzle with a straight bridge, open lower jaw
    add_sphere(bm, head, (0, 0.02, 0.118), (0.113, 0.14, 0.112))
    add_sphere(bm, head, (0, 0.09, 0.125), (0.08, 0.085, 0.06))
    for centre, radii in UPPER_JAW + LOWER_JAW:
        add_sphere(bm, head, centre, radii)
    add_chain(bm, head, (0, 0.14, 0.112), (0, 0.315, 0.09), 0.028, 0.02, 7)
    # Cheekbones, jaw muscles, angry brows, ears and cheek ruff
    for sx in (-1, 1):
        add_sphere(bm, head, (sx * 0.068, 0.075, 0.088), (0.04, 0.07, 0.038))
        add_sphere(bm, head, (sx * 0.06, 0.07, 0.035), (0.036, 0.06, 0.048))
        add_chain(bm, head, (sx * 0.02, 0.15, 0.150), (sx * 0.09, 0.10, 0.178), 0.02, 0.018, 5)
        add_cone(bm, head @ Vector((sx * 0.07, -0.02, 0.19)), head @ Vector((sx * 0.105, -0.045, 0.37)), 0.05, segments=10)
        for end in ((0.20, -0.05, 0.02), (0.18, -0.10, 0.08), (0.17, -0.02, -0.04), (0.15, -0.12, -0.02)):
            add_cone(bm, head @ Vector((sx * 0.08, 0.0, 0.07)), head @ Vector((sx * end[0], end[1], end[2])), 0.045)

    # Thick neck, shoulder hump and a bushy drooping tail
    add_chain(bm, world, bone["neck_01"].head, head @ Vector((0, 0.0, 0.06)), 0.10, 0.10, 5)
    add_sphere(bm, world, bone["neck_01"].head + Vector((0, 0.07, -0.04)), (0.21, 0.13, 0.15))
    for i in range(26):
        t = i / 25
        point = tail_point(tail_root, t)
        radius = 0.05 + 0.065 * math.sin(t**0.7 * math.pi) + 0.02 * (1 - t)
        add_sphere(bm, world, point, (radius, radius, radius))
        for _ in range(3):
            out = Vector((random.uniform(-1, 1), 0.6, random.uniform(-1, 0.6))).normalized()
            add_cone(bm, point, point + out * (radius + 0.07), radius * 0.6)

    # Shaggy fur tufts sprouting from the body, biased to hang down and back
    names = [b.name for b in bone]
    heads = np.array([b.head for b in bone])
    spans = np.array([b.tail - b.head for b in bone])
    coords = np.array([v.co for v in body.data.vertices])
    # Distance from every vertex to every bone segment picks the body region each vertex belongs to
    offsets = coords[:, None, :] - heads[None, :, :]
    along = np.clip((offsets * spans).sum(-1) / (spans * spans).sum(-1), 0.0, 1.0)
    nearest = np.linalg.norm(offsets - along[..., None] * spans, axis=-1).argmin(axis=1)
    candidates, weights = [], []
    for v, bone_index in zip(body.data.vertices, nearest):
        weight = next((w for key, w in TUFT_WEIGHTS.items() if names[bone_index].startswith(key)), 0.0)
        if weight:
            candidates.append(v)
            weights.append(weight)
    for v in random.choices(candidates, weights=weights, k=TUFT_COUNT):
        direction = (v.normal * 0.6 + Vector((0, 0.15, -0.55))).normalized()
        length = random.uniform(0.05, 0.11)
        add_cone(bm, v.co - direction * 0.02, v.co + direction * length, random.uniform(0.018, 0.03))
    return bm


def add_teeth(bm, teeth, spheres, side):
    """Plant both rows of a jaw's teeth on its top (side 1) or underside (side -1); return the gum line they stand on."""
    lean = Vector((0, 0.3, 1)).normalized() if side > 0 else Vector((0, 0, -1))  # the lower jaw slopes down toward its tip
    gum_line = []
    for sx in (-1, 1):
        seats = []
        for x, y, length, radius, flat in teeth:
            seat = Vector((sx * x, y, jaw_surface(spheres, x, y, side)))
            axis = (lean + Vector((sx * 0.08, 0, 0))).normalized()
            curl = Vector((0, -0.3 * length, 0)) if length > 0.03 else Vector()  # the long canines hook backward
            # A root sunk into the jaw, then a crown tapering to a point
            crown = (0.3, 0.55, 0.78, 0.92, 1.0)
            points = [seat - axis * 0.012, seat] + [seat + axis * length * t + curl * t * t for t in crown]
            radii = [radius, radius] + [radius * (1 - t) ** 0.75 for t in crown]
            add_loft(bm, points, radii, across=(1, 0, 0) if y > 0.303 else (0, 1, 0), flat=flat)
            seats.append(seat - Vector((0, 0, side * 0.002)))
        gum_line += seats[::sx]
    return gum_line


def add_eye(bm, sx):
    """Add one eyeball and the lids that leave an almond-shaped opening over it; return the faces of its pupil and rim."""
    centre = Vector((sx * EYE[0], EYE[1], EYE[2]))
    gaze = Vector((sx * 0.45, 1, -0.05)).normalized()
    outer = Vector((sx, 0, 0))
    outer = (outer - gaze * outer.dot(gaze)).normalized()
    up = outer.cross(gaze) * sx
    # Slant the eye so its outer corner sits higher than the inner one
    slant = math.radians(15)
    outer, up = outer * math.cos(slant) + up * math.sin(slant), up * math.cos(slant) - outer * math.sin(slant)
    turn = Matrix.Translation(centre) @ gaze.to_track_quat("Z", "Y").to_matrix().to_4x4()

    ball = bmesh.ops.create_uvsphere(bm, u_segments=20, v_segments=20, radius=0.017, matrix=turn)["verts"]
    dark = []
    for face in {f for v in ball for f in v.link_faces}:
        angle = math.degrees((face.calc_center_median() - centre).angle(gaze))
        if angle < 16 or angle > 50:
            dark.append(face)

    # The lids are a shell around the ball, wrapped around the line between the corners of the eye and
    # left open in front in an almond that is widest in the middle and closes toward each corner
    radius, along, around = 0.0183, 14, 16
    rows = []
    for i in range(along + 1):
        across = math.pi * (0.04 + 0.92 * i / along)
        gape = max(0.0, 1 - ((across - math.pi / 2) / math.radians(62)) ** 2)
        low, high = math.radians(22) * gape, 2 * math.pi - math.radians(19) * gape
        turns = [low + (high - low) * k / around for k in range(around + 1)]
        rows.append([bm.verts.new(centre + radius * (outer * math.cos(across) + (gaze * math.cos(t) + up * math.sin(t)) * math.sin(across)))
                     for t in turns])
    lid_faces = [bm.faces.new((a[k], b[k], b[k + 1], a[k + 1])[::sx]) for a, b in zip(rows, rows[1:]) for k in range(around)]
    return dark, lid_faces


def build_details(rig, head):
    """Return an object with the unfused details: teeth, gums, tongue, eyes, nose and claws."""
    bm = bmesh.new()
    bone = rig.pose.bones
    here = Matrix.Identity(4)
    jaw_side = bm.verts.layers.int.new("jaw_side")  # 1 for parts fixed to the skull, 2 for parts that swing with the jaw
    done = [0, 0]

    def tag(material_index, side=0):
        """Give everything added since the last call its material and, for parts of the face, the jaw it follows."""
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        for face in bm.faces[done[0]:]:
            face.material_index = material_index
            face.smooth = True
        for vert in bm.verts[done[1]:]:
            vert[jaw_side] = side
        done[:] = len(bm.faces), len(bm.verts)

    # The face is modelled in the head frame and moved into place afterwards
    upper_gums = add_teeth(bm, UPPER_TEETH, UPPER_JAW, -1)
    tag(0, 1)
    lower_gums = add_teeth(bm, LOWER_TEETH, LOWER_JAW, 1)
    tag(0, 2)
    add_loft(bm, upper_gums, [0.006] * len(upper_gums), across=(0, 0, 1), segments=6)
    tag(4, 1)
    add_loft(bm, lower_gums, [0.006] * len(lower_gums), across=(0, 0, 1), segments=6)
    tag(4, 2)

    # Tongue lying in the lower jaw with its tip just behind the front teeth
    spine = [(0.125, 0.024), (0.16, 0.027), (0.2, 0.026), (0.24, 0.022), (0.268, 0.016), (0.283, 0.009), (0.292, 0)]
    add_loft(bm, [(0, y, jaw_surface(LOWER_JAW, 0, y, 1) + 0.004) for y, _ in spine], [r for _, r in spine], flat=0.3, segments=10)
    tag(5, 2)

    # Eyes: amber iris, black pupil and rim, matt black lids
    for sx in (-1, 1):
        dark, lid_faces = add_eye(bm, sx)
        tag(1, 1)
        for face in dark:
            face.material_index = 2
        for face in lid_faces:
            face.material_index = 6

    # Nose: a broad pad with a lobe around each nostril
    add_sphere(bm, here, NOSE, (0.019, 0.013, 0.013), detail=2)
    for sx in (-1, 1):
        add_sphere(bm, here, (sx * 0.0095, NOSE[1] + 0.001, NOSE[2] - 0.004), (0.01, 0.012, 0.01), detail=2)
    tag(3, 1)
    for sx in (-1, 1):
        add_sphere(bm, here, (sx * 0.009, NOSE[1] + 0.0105, NOSE[2] - 0.003), (0.0045, 0.003, 0.0035))
    tag(2, 1)
    bm.transform(head)

    # Claws on every fingertip and four per foot
    for side in ("l", "r"):
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            pb = bone[f"{finger}_03_{side}"]
            direction = (pb.tail - pb.head).normalized()
            add_cone(bm, pb.tail - direction * 0.015, pb.tail + direction * 0.06 - Z_AXIS * 0.012, 0.012, segments=8)
        pb = bone[f"ball_{side}"]
        direction = (pb.tail - pb.head).normalized()
        lateral = direction.cross(Z_AXIS).normalized()
        for offset in (-0.04, -0.014, 0.014, 0.04):
            base = pb.tail + lateral * offset - direction * 0.02
            add_cone(bm, base, base + direction * 0.07 + lateral * offset * 0.3 - Z_AXIS * 0.01, 0.013, segments=8)
    tag(0)

    details = bmesh_to_object(bm, "Details")
    details.data.materials.append(make_material("Ivory", (0.9, 0.85, 0.72), roughness=0.35))
    details.data.materials.append(make_material("Iris", (0.9, 0.45, 0.04), roughness=0.15, emission=0.9))
    details.data.materials.append(make_material("Pupil", (0.004, 0.004, 0.004), roughness=0.1))
    details.data.materials.append(make_material("Nose", (0.015, 0.014, 0.014), roughness=0.4))
    details.data.materials.append(make_material("Gums", (0.2, 0.05, 0.055), roughness=0.3))
    details.data.materials.append(make_material("Tongue", (0.5, 0.13, 0.15), roughness=0.3))
    details.data.materials.append(make_material("Lids", (0.02, 0.018, 0.016), roughness=0.8))
    return details


def fuse_and_fur(body, parts):
    """Voxel-fuse the body with the wolf parts, roughen the surface and reduce it to the triangle budget."""
    creature = join_objects([body, parts], "Werewolf")
    creature.data.materials.clear()
    remesh = creature.modifiers.new("Remesh", "REMESH")
    remesh.mode = "VOXEL"
    remesh.voxel_size = VOXEL_SIZE
    apply_modifier(creature, remesh)
    smooth = creature.modifiers.new("Smooth", "SMOOTH")
    smooth.factor, smooth.iterations = 0.6, 4
    apply_modifier(creature, smooth)
    texture = bpy.data.textures.new("FurNoise", "CLOUDS")
    texture.noise_scale = 0.03
    displace = creature.modifiers.new("Fur", "DISPLACE")
    displace.texture, displace.strength, displace.mid_level = texture, 0.014, 0.5
    apply_modifier(creature, displace)
    triangles = sum(len(p.vertices) - 2 for p in creature.data.polygons)
    decimate = creature.modifiers.new("Decimate", "DECIMATE")
    decimate.ratio = min(1.0, TARGET_TRIANGLES / triangles)
    apply_modifier(creature, decimate)
    for poly in creature.data.polygons:
        poly.use_smooth = True
    return creature


def paint_fur(creature, rig, head):
    """Vertex-colour the fur: dark coat, paler chest, belly and lower muzzle, darker paws and mottling."""
    mesh = creature.data
    bone = rig.pose.bones
    to_head = head.inverted()
    head_axes = to_head.to_3x3().normalized()
    eyes, nose = [Vector((sx * EYE[0], EYE[1], EYE[2])) for sx in (-1, 1)], Vector(NOSE)
    pelvis_z, neck_z = bone["pelvis"].head.z, bone["neck_01"].head.z
    dark, pale, black = Vector((0.075, 0.055, 0.04)), Vector((0.40, 0.32, 0.23)), Vector((0.02, 0.018, 0.016))
    flesh = Vector((0.13, 0.03, 0.035))
    paws = [bone[n].head.copy() for n in ("hand_l", "hand_r", "foot_l", "foot_r")]

    colors = np.empty((len(mesh.vertices), 4), dtype=np.float32)
    for v in mesh.vertices:
        co, local = v.co, to_head @ v.co
        belly = max(0.0, -v.normal.y) * (abs(co.x) < 0.24) * (pelvis_z - 0.1 < co.z < neck_z)
        on_head = abs(local.x) < 0.13 and local.z > -0.10 and local.length < 0.45
        # Paler cheeks, lower muzzle and chin under a dark forehead and bridge
        muzzle = on_head * smoothstep(local.y, 0.0, 0.08) * (1 - smoothstep(local.z, 0.045, 0.10))
        paleness = min(1.0, max(belly * 0.9, muzzle * 0.8))
        color = dark.lerp(pale, paleness) * (0.75 + 0.5 * noise.noise(co * 9.0) + 0.25 * noise.noise(co * 40.0))
        paw = min((co - p).length for p in paws)
        color = color.lerp(black, max(0.0, 1.0 - paw / 0.16) * 0.8)
        if on_head:
            # Black lips, eye rims and skin around the nose, and dark flesh inside the mouth
            lip, inside = mouth_mask(local, (head_axes @ v.normal).z)
            rim = 1 - smoothstep(min((local - e).length for e in eyes), 0.02, 0.04)
            color = color.lerp(black, max(lip, rim, 1 - smoothstep((local - nose).length, 0.025, 0.05))).lerp(flesh, inside)
        colors[v.index] = (*[max(0.0, c) for c in color], 1.0)

    attribute = mesh.color_attributes.new("Fur", "BYTE_COLOR", "CORNER")
    corner_vertices = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", corner_vertices)
    attribute.data.foreach_set("color", colors[corner_vertices].ravel())
    mesh.color_attributes.active_color = attribute
    mesh.materials.append(make_material("Fur", (0.1, 0.08, 0.07), roughness=0.85, vertex_color="Fur"))
    return colors


def make_fur_texture(size=512, columns=4, strands_per_column=26):
    """Paint a texture of hair locks: fine strands that start spread out at the bottom and gather toward the tip."""
    rng = np.random.default_rng(3)
    pixels = np.zeros((size, size, 4), dtype=np.float32)
    column_width = size / columns
    for column in range(columns):
        left, right = int(column * column_width), int((column + 1) * column_width) - 1
        tip_x = left + column_width * rng.uniform(0.4, 0.6)
        for strand in range(strands_per_column):
            x0 = left + (strand + 0.5 + rng.uniform(-0.4, 0.4)) / strands_per_column * column_width
            rows = int(rng.uniform(0.5, 1.0) * size)
            gather = rng.uniform(0.5, 0.95)
            half = rng.uniform(1.3, 2.6)
            shade = rng.uniform(0.72, 1.0)
            for row in range(rows):
                t = row / size
                centre = x0 + (tip_x - x0) * gather * t**1.5
                width = half * (1 - row / rows) ** 0.5 + 0.3
                lo, hi = max(left, round(centre - width)), min(right, round(centre + width))
                if hi >= lo:
                    brightness = shade * (0.8 + 0.2 * t)
                    pixels[row, lo:hi + 1] = (brightness, brightness, brightness, 1.0)

    image = bpy.data.images.new("FurStrands", size, size, alpha=True)
    image.pixels.foreach_set(pixels.ravel())
    os.makedirs(os.path.join(OUT_DIR, "cache"), exist_ok=True)
    image.filepath_raw = os.path.join(OUT_DIR, "cache", "fur_strands.png")
    image.file_format = "PNG"
    image.save()
    return image, columns


def make_fur_card_material(image):
    """Strand texture tinted by the coat's vertex colours, with hard alpha cut-out and no backface culling."""
    mat = bpy.data.materials.new("FurCards")
    mat.use_nodes = True
    tree = mat.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.8
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = image
    coat = tree.nodes.new("ShaderNodeVertexColor")
    coat.layer_name = "Fur"
    tint = tree.nodes.new("ShaderNodeMix")
    tint.data_type, tint.blend_type = "RGBA", "MULTIPLY"
    tint.inputs[0].default_value = 1.0
    tree.links.new(texture.outputs["Color"], tint.inputs[6])
    tree.links.new(coat.outputs["Color"], tint.inputs[7])
    tree.links.new(tint.outputs[2], bsdf.inputs["Base Color"])
    cutout = tree.nodes.new("ShaderNodeMath")
    cutout.operation = "ROUND"
    tree.links.new(texture.outputs["Alpha"], cutout.inputs[0])
    tree.links.new(cutout.outputs[0], bsdf.inputs["Alpha"])
    mat.use_backface_culling = False
    return mat


def build_fur_cards(creature, colors, rig, head):
    """Cover the coat with small textured hair cards that lie along the fur's flow direction."""
    mesh = creature.data
    mesh.calc_loop_triangles()
    count = len(mesh.loop_triangles)
    triangles = np.empty(count * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", triangles)
    triangles = triangles.reshape(-1, 3)
    areas = np.empty(count, dtype=np.float32)
    mesh.loop_triangles.foreach_get("area", areas)
    coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", coords)
    coords = coords.reshape(-1, 3)
    normals = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("normal", normals)
    normals = normals.reshape(-1, 3)

    rng = np.random.default_rng(5)
    image, columns = make_fur_texture()

    bone = rig.pose.bones
    to_head = head.inverted()
    head_axes = to_head.to_3x3().normalized()
    # The short fur of the muzzle needs many more cards for the same cover, so sample it more densely
    centres = coords[triangles].mean(axis=1) @ np.array(to_head.to_3x3(), dtype=np.float32).T + np.array(to_head.translation, dtype=np.float32)
    on_muzzle = (centres[:, 1] > 0.1) & (np.abs(centres[:, 0]) < 0.1) & (centres[:, 2] > -0.09) & (np.linalg.norm(centres, axis=1) < 0.45)
    density = areas * np.where(on_muzzle, 5.0, 1.0)
    picks = rng.choice(count, size=FUR_CARD_COUNT, p=density / density.sum())
    weights = rng.dirichlet((1, 1, 1), size=FUR_CARD_COUNT)
    head_back = (head.to_3x3() @ Vector((0, -1, -0.35))).normalized()
    hang = Vector((0, 0.35, -1)).normalized()
    tail_flow = Vector((0, 0.8, -0.45)).normalized()
    pelvis = bone["pelvis"].head.copy()
    paws = [bone[n].head.copy() for n in ("hand_l", "hand_r", "foot_l", "foot_r")]
    eyes, nose = [Vector((sx * EYE[0], EYE[1], EYE[2])) for sx in (-1, 1)], Vector(NOSE)

    def style(p, n):
        """Return (strand length, flow direction) for a point on the coat, or None to leave it bare."""
        local = to_head @ p
        muzzle = local.y > 0.1 and abs(local.x) < 0.1 and local.z > -0.09
        if (local.z > 0.0 or muzzle) and local.length < 0.45:
            if mouth_mask(local, (head_axes @ n).z)[0] > 0.3 or (local - nose).length < 0.026 or min((local - e).length for e in eyes) < 0.024:
                return None
            if local.z > 0.26:
                return rng.uniform(0.02, 0.03), head_back
            # Fur shortens from the cheeks and forehead down the muzzle
            # Fur shortens from the cheeks and forehead down the muzzle, and around the eyes so it does not hide them
            eye = min((local - e).length for e in eyes)
            return rng.uniform(0.8, 1.2) * (0.05 - 0.034 * smoothstep(local.y, 0.08, 0.17)) * (0.3 + 0.7 * smoothstep(eye, 0.03, 0.07)), head_back
        if min((p - paw).length for paw in paws) < 0.17:
            return rng.uniform(0.02, 0.035), hang
        if p.y > pelvis.y + 0.2:
            return rng.uniform(0.07, 0.11), tail_flow
        if n.y > 0.3 and p.z > pelvis.z + 0.15:
            return rng.uniform(0.07, 0.11), hang
        return rng.uniform(0.045, 0.08), hang

    verts, faces, uvs, tints, card_normals, roots = [], [], [], [], [], []

    def add_card(p, n, length, flow, color):
        flow = flow + Vector(rng.uniform(-0.25, 0.25, 3))
        tangent = flow - n * flow.dot(n)
        if tangent.length < 1e-4:
            tangent = n.orthogonal()
        fine = length < 0.024  # the muzzle's fur is narrow and lies close to the skin
        direction = (tangent.normalized() + n * (0.25 if fine else 0.5)).normalized()
        across = direction.cross(n).normalized()
        width = min(0.04, max(0.009 if fine else 0.014, length * 0.55))
        brightness = rng.uniform(0.9, 1.2)
        column = int(rng.integers(columns))
        shade_normal = (n * 0.75 + direction * 0.25).normalized()
        base = len(verts)
        for row, t in enumerate((0.0, 1.0) if fine else (0.0, 0.5, 1.0)):
            centre = p - n * 0.004 + direction * length * t - n * length * 0.18 * t * t
            half = across * width * 0.5 * (1 - 0.45 * t)
            for side, u in ((-1, 0.0), (1, 1.0)):
                verts.append(centre + half * side)
                uvs.append(((column + u) / columns, t))
                tints.append((*(color * brightness * (1.0 + 0.5 * t)), 1.0))
                card_normals.append(shade_normal)
                roots.append(p)
            if row:
                faces.append((base + 2 * row - 2, base + 2 * row - 1, base + 2 * row + 1, base + 2 * row))

    for tri, w in zip(picks, weights):
        corner = triangles[tri]
        p, n = Vector(w @ coords[corner]), Vector(w @ normals[corner]).normalized()
        styled = style(p, n)
        if styled is None:
            continue
        length, flow = styled
        # Short fur covers less per card, so plant a few cards around each sample point there
        for _ in range(4 if length < 0.024 else 3 if length < 0.036 else 1):
            add_card(p + Vector(rng.uniform(-0.012, 0.012, 3)) * (length < 0.036), n, length, flow, Vector(w @ colors[corner][:, :3]))

    fur_mesh = bpy.data.meshes.new("WerewolfFur")
    fur_mesh.from_pydata(verts, [], faces)
    corner_vertices = np.empty(len(fur_mesh.loops), dtype=np.int32)
    fur_mesh.loops.foreach_get("vertex_index", corner_vertices)
    fur_mesh.uv_layers.new(name="UVMap").data.foreach_set("uv", np.array(uvs, dtype=np.float32)[corner_vertices].ravel())
    attribute = fur_mesh.color_attributes.new("Fur", "BYTE_COLOR", "CORNER")
    attribute.data.foreach_set("color", np.clip(np.array(tints, dtype=np.float32), 0, 1)[corner_vertices].ravel())
    fur_mesh.polygons.foreach_set("use_smooth", [True] * len(fur_mesh.polygons))
    # Shade the cards mostly like the skin underneath so they read as a soft coat, not flat blades
    fur_mesh.normals_split_custom_set_from_vertices(card_normals)
    fur_mesh.materials.append(make_fur_card_material(image))
    fur = bpy.data.objects.new("WerewolfFur", fur_mesh)
    bpy.context.collection.objects.link(fur)
    return fur, np.array(roots, dtype=np.float32)


def prepare_rig(rig, tail_root, head):
    """Make the creature's stance the rig's rest pose and add a chain of tail bones and a jaw bone."""
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode="EDIT")
    parent = rig.data.edit_bones["pelvis"]
    parent.use_connect = False
    for i in range(TAIL_BONES):
        eb = rig.data.edit_bones.new(f"tail_{i + 1:02d}")
        eb.head, eb.tail = tail_point(tail_root, i / TAIL_BONES), tail_point(tail_root, (i + 1) / TAIL_BONES)
        eb.parent, eb.use_connect = parent, i > 0
        parent = eb
    jaw = rig.data.edit_bones.new("jaw")
    jaw.head, jaw.tail = head @ Vector(JAW_HINGE), head @ Vector(JAW_TIP)
    jaw.parent = rig.data.edit_bones["head"]
    bpy.ops.object.mode_set(mode="OBJECT")


def bone_radius(name):
    base = name[:-2] if name.endswith(("_l", "_r")) else name
    return BONE_RADIUS.get("tail" if base.startswith("tail_") else base)


def skin_weights(points, rig, head, jaw_side=None):
    """Return (bone names, weights per point) from each point's distance to the surface around every bone."""
    bones = [b for b in rig.data.bones if bone_radius(b.name)]
    names = [b.name for b in bones]
    heads = np.array([b.head_local for b in bones], dtype=np.float32)
    spans = np.array([b.tail_local - b.head_local for b in bones], dtype=np.float32)
    radii = np.array([bone_radius(n) for n in names], dtype=np.float32)
    weights = np.empty((len(points), len(bones)), dtype=np.float32)
    for start in range(0, len(points), 20000):
        offsets = points[start:start + 20000, None, :] - heads[None, :, :]
        along = np.clip((offsets * spans).sum(-1) / (spans * spans).sum(-1), 0.0, 1.0)
        dist = np.linalg.norm(offsets - along[..., None] * spans, axis=-1) - radii
        weights[start:start + 20000] = np.exp(-(dist - dist.min(axis=1, keepdims=True)) / SKIN_BLEND)

    # The whole wolf head (muzzle, ears, ruff) follows the head bone, blending in above the skull base
    to_head = np.array(head.inverted(), dtype=np.float32)
    local = points @ to_head[:3, :3].T + to_head[:3, 3]
    upper = [i for i, n in enumerate(names) if n in ("head", "neck_01", "spine_03", "clavicle_l", "clavicle_r")]
    blend = np.clip((local[:, 2] + 0.02) / 0.08, 0.0, 1.0) * (np.linalg.norm(local, axis=1) < 0.5) * np.isin(weights.argmax(axis=1), upper)
    weights *= (1.0 - blend)[:, None]
    weights[:, names.index("head")] += blend

    # The lower jaw swings on its own bone: everything in the muzzle below the line where the jaws part
    below = np.clip((parting(local[:, 1]) - local[:, 2]) / 0.008 + 0.5, 0.0, 1.0)
    hinge = np.clip((local[:, 1] - 0.10) / 0.07, 0.0, 1.0)
    jaw = below * hinge * (np.abs(local[:, 0]) < 0.09) * (local[:, 2] > -0.08) * (local[:, 1] < 0.40)
    names.append("jaw")
    weights = np.hstack([weights * (1.0 - jaw)[:, None], jaw[:, None].astype(np.float32)])
    # Teeth, tongue, eyes and nose each follow the skull or the jaw as a whole
    if jaw_side is not None:
        for side, name in ((1, "head"), (2, "jaw")):
            weights[jaw_side == side] = np.eye(len(names), dtype=np.float32)[names.index(name)]

    # Keep the four strongest bones per point, as glTF skinning expects
    weights[weights < np.partition(weights, -4, axis=1)[:, -4:-3]] = 0.0
    return names, weights / weights.sum(axis=1, keepdims=True)


def bind(obj, rig, names, weights):
    """Skin an object to the rig with vertex groups, writing weights in sixteenths so they can be added in batches."""
    steps = np.round(weights * 16).astype(int)
    obj.vertex_groups.clear()  # drop the human's leftover groups so the new ones keep the bone names
    for column, name in enumerate(names):
        group = obj.vertex_groups.new(name=name)
        for step in np.unique(steps[:, column]):
            if step:
                group.add(np.nonzero(steps[:, column] == step)[0].tolist(), step / 16, "REPLACE")
    obj.parent = rig
    obj.modifiers.new("Armature", "ARMATURE").object = rig


def dance_pose(frame):
    """Return (pelvis offset, armature-space rotations with parents first, foot lifts) for one frame of the dance."""
    beat = 2 * math.pi * frame / 12
    sway = math.sin(beat / 2)  # hips go side to side once every two beats
    dip = 0.5 + 0.5 * math.cos(beat)  # 1 on every beat, 0 between beats
    twist = math.sin(beat / 4)
    howl = math.sin(math.pi * max(0, frame - 72) / 24) ** 2  # throw the head back over the last two beats
    left, right = 0.5 + 0.5 * sway, 0.5 - 0.5 * sway
    offset = Vector((0.07 * sway, 0.0, 0.02 - 0.08 * dip))
    rotations = [
        ("pelvis", "Y", 4 * sway), ("pelvis", "Z", -8 * twist),
        ("spine_01", "Y", -5 * sway), ("spine_01", "X", 5 * dip - 8 * howl),
        ("spine_02", "Y", -5 * sway), ("spine_02", "X", 4 * dip - 8 * howl),
        ("spine_03", "Z", 12 * twist), ("neck_01", "X", 4 * dip - 10 * howl),
        ("head", "Y", 7 * sway), ("head", "X", 8 * dip - 4 - 30 * howl),
        ("upperarm_l", "X", -55 * left), ("lowerarm_l", "X", 25 * left - 18 * dip),
        ("upperarm_r", "X", -55 * right), ("lowerarm_r", "X", 25 * right - 18 * dip),
    ]
    rotations += [(f"tail_{i + 1:02d}", "Z", 14 * math.sin(beat - 0.7 * i)) for i in range(TAIL_BONES)]
    # The foot on the unweighted side steps off the ground
    lifts = {"l": 0.09 * max(0.0, -sway) ** 2, "r": 0.09 * max(0.0, sway) ** 2}
    return offset, rotations, lifts


def animate_dance(rig):
    """Key a looping dance on the rig; feet are held by temporary IK targets and then baked to plain keyframes."""
    bones = rig.pose.bones
    for pb in bones:
        pb.rotation_mode = "QUATERNION"
    animated = sorted({name for name, _, _ in dance_pose(0)[1]})

    # Work out every frame's pose first, because inserting keys while posing would let the animation overwrite the pose
    poses = []
    for frame in range(DANCE_FRAMES + 1):
        for name in animated:
            bones[name].matrix_basis = Matrix.Identity(4)
        bpy.context.view_layer.update()
        offset, rotations, lifts = dance_pose(frame)
        bones["pelvis"].matrix = Matrix.Translation(offset) @ bones["pelvis"].matrix
        bpy.context.view_layer.update()
        for name, axis, degrees in rotations:
            pb = bones[name]
            pivot = Matrix.Translation(pb.matrix.translation)
            pb.matrix = pivot @ Matrix.Rotation(math.radians(degrees), 4, axis) @ pivot.inverted() @ pb.matrix
            bpy.context.view_layer.update()
        pose = {name: (bones[name].location.copy(), bones[name].rotation_quaternion.copy()) for name in animated}
        # Keep quaternions on the same side as the previous frame so interpolation takes the short way
        for name, (_, quat) in pose.items():
            if poses and quat.dot(poses[-1][0][name][1]) < 0:
                quat.negate()
        poses.append((pose, lifts))
    for name in animated:
        bones[name].matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()

    targets = {}
    for side in ("l", "r"):
        foot = bones[f"foot_{side}"]
        target = bpy.data.objects.new(f"FootTarget_{side}", None)
        bpy.context.collection.objects.link(target)
        target.matrix_world = rig.matrix_world @ foot.matrix
        ik = bones[f"calf_{side}"].constraints.new("IK")
        ik.target, ik.chain_count = target, 2
        foot.constraints.new("COPY_ROTATION").target = target
        targets[side] = (target, target.location.copy())

    for frame, (pose, lifts) in enumerate(poses):
        for name, (location, quat) in pose.items():
            pb = bones[name]
            pb.location, pb.rotation_quaternion = location, quat
            pb.keyframe_insert("location", frame=frame)
            pb.keyframe_insert("rotation_quaternion", frame=frame)
        for side, (target, rest) in targets.items():
            target.location = rest + Z_AXIS * lifts[side]
            target.keyframe_insert("location", frame=frame)

    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 0, DANCE_FRAMES
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.nla.bake(frame_start=0, frame_end=DANCE_FRAMES, only_selected=False, visual_keying=True,
                     clear_constraints=True, use_current_action=True, bake_types={"POSE"})
    bpy.ops.object.mode_set(mode="OBJECT")
    dance = rig.animation_data.action
    dance.name = "Dance"
    for target, _ in targets.values():
        bpy.data.objects.remove(target)
    for action in list(bpy.data.actions):
        if action != dance:
            bpy.data.actions.remove(action)


def build_werewolf():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    body, rig = posed_human()
    head = head_frame(rig)
    tail_root = rig.pose.bones["pelvis"].head + Vector((0, 0.10, 0.02))
    parts = bmesh_to_object(build_wolf_parts(body, rig, head, tail_root), "WolfParts")
    details = build_details(rig, head)
    creature = fuse_and_fur(body, parts)
    colors = paint_fur(creature, rig, head)
    fur, fur_roots = build_fur_cards(creature, colors, rig, head)
    werewolf = join_objects([creature, details], "Werewolf")
    prepare_rig(rig, tail_root, head)

    # Stand the model, its fur and its rig on the ground plane
    lift = -min(v.co.z for v in werewolf.data.vertices)
    werewolf.location.z = fur.location.z = rig.location.z = lift
    for obj in (werewolf, fur, rig):
        obj.select_set(True)
    bpy.context.view_layer.objects.active = werewolf
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    head = Matrix.Translation((0, 0, lift)) @ head

    # Animate before skinning so posing the rig does not have to deform the meshes on every step
    animate_dance(rig)
    coords = np.empty(len(werewolf.data.vertices) * 3, dtype=np.float32)
    werewolf.data.vertices.foreach_get("co", coords)
    jaw_side = np.empty(len(werewolf.data.vertices), dtype=np.int32)
    werewolf.data.attributes["jaw_side"].data.foreach_get("value", jaw_side)
    bind(werewolf, rig, *skin_weights(coords.reshape(-1, 3), rig, head, jaw_side))
    bind(fur, rig, *skin_weights(fur_roots + np.array((0, 0, lift), dtype=np.float32), rig, head))
    return werewolf, head


def export_and_render(werewolf, head):
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT_DIR, "werewolf.blend"))
    bpy.ops.export_scene.gltf(
        filepath=os.path.join(OUT_DIR, "werewolf.glb"),
        export_format="GLB",
        export_apply=True,
        export_cameras=False,
        export_lights=False,
        export_draco_mesh_compression_enable=True,
        export_animations=True,
        export_optimize_animation_size=True,
    )

    # Preview renders: moonlit key light, warm rim light, dark ground
    scene = bpy.context.scene
    scene.frame_set(STILL_FRAME)
    bpy.ops.mesh.primitive_plane_add(size=30)
    bpy.context.active_object.data.materials.append(make_material("Ground", (0.05, 0.06, 0.05), roughness=0.95))
    for name, energy, color, angles in (("Moon", 3.5, (0.75, 0.85, 1.0), (55, 0, 30)), ("Rim", 2.5, (1.0, 0.6, 0.35), (70, 0, 200))):
        light = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
        light.data.energy, light.data.color = energy, color
        light.rotation_euler = [math.radians(a) for a in angles]
        scene.collection.objects.link(light)
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.03, 0.045, 0.08, 1.0)
    scene.world = world
    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.transparent_max_bounces = 32
    scene.render.resolution_x, scene.render.resolution_y = 1000, 1100
    height = max(v.co.z for v in werewolf.data.vertices)
    head_pos = Vector((0, head.translation.y - 0.12, height - 0.25))
    for filename, position, target in (
        ("werewolf_preview.png", Vector((2.3, -3.6, 1.5)), Vector((0, 0, height * 0.5))),
        ("werewolf_head_preview.png", head_pos + Vector((0.55, -0.85, 0.12)), head_pos),
    ):
        cam.location = position
        cam.rotation_euler = (target - position).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = os.path.join(OUT_DIR, filename)
        bpy.ops.render.render(write_still=True)


model, head_matrix = build_werewolf()
export_and_render(model, head_matrix)
print(f"WEREWOLF_DONE triangles={sum(len(p.vertices) - 2 for o in bpy.data.objects if o.name.startswith('Werewolf') for p in o.data.polygons)} height={max(v.co.z for v in model.data.vertices):.2f}")
