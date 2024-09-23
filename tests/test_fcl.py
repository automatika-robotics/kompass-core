import json

import fcl
import numpy as np


def print_collision_result(o1_name, o2_name, result):
    print(f"Collision between {o1_name} and {o2_name}:")
    print("-" * 30)
    print(f"Collision?: {result.is_collision}")
    print(f"Number of contacts: {len(result.contacts)}")
    print("")
    test_name = f"{o1_name}_{o2_name}_pairwise_collision"
    test_result = {
        "collision": result.is_collision,
        "num_contacts": len(result.contacts),
    }
    return test_name, test_result


def print_continuous_collision_result(o1_name, o2_name, result):
    print(f"Continuous collision between {o1_name} and {o2_name}:")
    print("-" * 30)
    print(f"Collision?: {result.is_collide}")
    print(f"Time of collision: {result.time_of_contact}")
    print("")
    test_name = f"{o1_name}_{o2_name}_continous_collision"
    test_result = {
        "collision": result.is_collide,
        "collision_time": result.time_of_contact,
    }
    return test_name, test_result


def print_distance_result(o1_name, o2_name, result):
    print(f"Distance between {o1_name} and {o2_name}:")
    print("-" * 30)
    print(f"Distance: {result.min_distance}")
    print("Closest Points:")
    print(result.nearest_points[0])
    print(result.nearest_points[1])
    print("")
    test_name = f"{o1_name}_{o2_name}_distance"
    test_result = {
        "distance": result.min_distance,
        "closest_point_1": {
            "x": result.nearest_points[0][0],
            "y": result.nearest_points[0][1],
            "z": result.nearest_points[0][2],
        },
        "closest_point_2": {
            "x": result.nearest_points[1][0],
            "y": result.nearest_points[1][1],
            "z": result.nearest_points[1][2],
        },
    }
    return test_name, test_result


def test_fcl(save_results: bool = False):
    # Create simple geometries
    box = fcl.Box(1.0, 2.0, 3.0)
    sphere = fcl.Sphere(4.0)
    cone = fcl.Cone(5.0, 6.0)
    cyl = fcl.Cylinder(2.0, 2.0)

    verts = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0],
    ])
    tris = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]])

    # Create mesh geometry
    mesh = fcl.BVHModel()
    mesh.beginModel(len(verts), len(tris))
    mesh.addSubModel(verts, tris)
    mesh.endModel()

    # =====================================================================
    # Pairwise collision checking
    # =====================================================================
    print("=" * 60)
    print("Testing Pairwise Collision Checking")
    print("=" * 60)
    print("")
    results = {}

    req = fcl.CollisionRequest(enable_contact=True)
    res = fcl.CollisionResult()

    fcl.collide(
        fcl.CollisionObject(box, fcl.Transform()),
        fcl.CollisionObject(cone, fcl.Transform()),
        req,
        res,
    )
    res_name, res_dict = print_collision_result("Box", "Cone", res)
    results[res_name] = res_dict

    assert res.is_collision is True

    fcl.collide(
        fcl.CollisionObject(box, fcl.Transform()),
        fcl.CollisionObject(cyl, fcl.Transform(np.array([6.0, 0.0, 0.0]))),
        req,
        res,
    )
    res_name, res_dict = print_collision_result("Box", "Cylinder", res)
    results[res_name] = res_dict

    assert res.is_collision is True

    fcl.collide(
        fcl.CollisionObject(mesh, fcl.Transform(np.array([0.0, 0.0, -1.0]))),
        fcl.CollisionObject(cyl, fcl.Transform()),
        req,
        res,
    )
    res_name, res_dict = print_collision_result("Box", "Mesh", res)
    results[res_name] = res_dict
    assert res.is_collision is True
    # =====================================================================
    # Pairwise distance checking
    # =====================================================================
    print("=" * 60)
    print("Testing Pairwise Distance Checking")
    print("=" * 60)
    print("")

    req = fcl.DistanceRequest(enable_nearest_points=True, enable_signed_distance=True)
    res = fcl.DistanceResult()

    fcl.distance(
        fcl.CollisionObject(box, fcl.Transform()),
        fcl.CollisionObject(cone, fcl.Transform()),
        req,
        res,
    )
    res_name, res_dict = print_distance_result("Box", "Cone", res)
    results[res_name] = res_dict

    assert res.min_distance < 0.0

    fcl.distance(
        fcl.CollisionObject(box, fcl.Transform()),
        fcl.CollisionObject(cyl, fcl.Transform(np.array([6.0, 0.0, 0.0]))),
        req,
        res,
    )
    res_name, res_dict = print_distance_result("Box", "Cylinder", res)
    results[res_name] = res_dict

    assert res.min_distance < 0.0

    fcl.distance(
        fcl.CollisionObject(box, fcl.Transform()),
        fcl.CollisionObject(box, fcl.Transform(np.array([1.01, 0.0, 0.0]))),
        req,
        res,
    )
    res_name, res_dict = print_distance_result("Box", "Box", res)
    results[res_name] = res_dict

    assert res.min_distance < 0.0

    # =====================================================================
    # Pairwise continuous collision checking
    # =====================================================================
    print("=" * 60)
    print("Testing Pairwise Continuous Collision Checking")
    print("=" * 60)
    print("")

    req = fcl.ContinuousCollisionRequest()
    res = fcl.ContinuousCollisionResult()

    fcl.continuousCollide(
        fcl.CollisionObject(box, fcl.Transform()),
        fcl.Transform(np.array([5.0, 0.0, 0.0])),
        fcl.CollisionObject(cyl, fcl.Transform(np.array([5.0, 0.0, 0.0]))),
        fcl.Transform(np.array([0.0, 0.0, 0.0])),
        req,
        res,
    )
    res_name, res_dict = print_continuous_collision_result("Box", "Cylinder", res)
    results[res_name] = res_dict

    assert res.is_collide is True

    # =====================================================================
    # Managed collision checking
    # =====================================================================
    print("=" * 60)
    print("Testing Managed Collision and Distance Checking")
    print("=" * 60)
    print("")
    objs1 = [
        fcl.CollisionObject(box, fcl.Transform(np.array([20, 0, 0]))),
        fcl.CollisionObject(sphere),
    ]
    objs2 = [fcl.CollisionObject(cone), fcl.CollisionObject(mesh)]
    objs3 = [fcl.CollisionObject(box), fcl.CollisionObject(sphere)]

    manager1 = fcl.DynamicAABBTreeCollisionManager()
    manager2 = fcl.DynamicAABBTreeCollisionManager()
    manager3 = fcl.DynamicAABBTreeCollisionManager()

    manager1.registerObjects(objs1)
    manager2.registerObjects(objs2)
    manager3.registerObjects(objs3)

    manager1.setup()
    manager2.setup()
    manager3.setup()

    # =====================================================================
    # Managed internal (n^2) collision checking
    # =====================================================================
    cdata = fcl.CollisionData()
    manager1.collide(cdata, fcl.defaultCollisionCallback)
    print(f"Collision within manager 1?: {cdata.result.is_collision}")
    print("")

    assert cdata.result.is_collision is False

    cdata = fcl.CollisionData()
    manager2.collide(cdata, fcl.defaultCollisionCallback)
    print(f"Collision within manager 2?: {cdata.result.is_collision}")
    print("")

    assert cdata.result.is_collision is True

    # =====================================================================
    # Managed internal (n^2) distance checking
    # =====================================================================
    ddata = fcl.DistanceData()
    manager1.distance(ddata, fcl.defaultDistanceCallback)
    print(f"Closest distance within manager 1?: {ddata.result.min_distance}")
    print("")

    ddata = fcl.DistanceData()
    manager2.distance(ddata, fcl.defaultDistanceCallback)
    print(f"Closest distance within manager 2?: {ddata.result.min_distance}")
    print("")

    # =====================================================================
    # Managed one to many collision checking
    # =====================================================================
    req = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
    rdata = fcl.CollisionData(request=req)

    manager1.collide(fcl.CollisionObject(mesh), rdata, fcl.defaultCollisionCallback)
    print(f"Collision between manager 1 and Mesh?: {rdata.result.is_collision}")
    print("Contacts:")
    for c in rdata.result.contacts:
        print(f"\tO1: {c.o1}, O2: {c.o2}")
    print("")

    assert rdata.result.is_collision is True

    # =====================================================================
    # Managed many to many collision checking
    # =====================================================================
    rdata = fcl.CollisionData(request=req)
    manager3.collide(manager2, rdata, fcl.defaultCollisionCallback)
    print(f"Collision between manager 2 and manager 3?: {rdata.result.is_collision}")
    print("Contacts:")
    for c in rdata.result.contacts:
        print(f"\tO1: {c.o1}, O2: {c.o2}")
    print("")

    assert rdata.result.is_collision is True

    # =====================================================================
    # Test pointcloud
    # =====================================================================

    x = np.random.random([10, 3])

    object1 = fcl.OcTree(0.01, points=x)
    print(f"pointcloud aabb center {object1.aabb_center}")

    object2 = fcl.Box(1, 1, 1)
    o2 = fcl.CollisionObject(object2)

    trans = np.array([2.0, 0.0, 0.0])
    o2.setTranslation(trans)

    o1 = fcl.CollisionObject(object1)

    req2 = fcl.DistanceRequest(enable_nearest_points=True, enable_signed_distance=True)
    res2 = fcl.DistanceResult()

    fcl.distance(
        o1,
        o2,
        req2,
        res2,
    )
    res_name, res_dict = print_distance_result("pointCloud", "Box", res2)
    results[res_name] = res_dict

    assert res2.min_distance > 0.0

    # with manager
    objects = [o1]
    manager1 = fcl.DynamicAABBTreeCollisionManager()
    manager1.registerObjects(objects)
    manager1.setup()

    cdata = fcl.CollisionData()
    manager1.collide(cdata, fcl.defaultCollisionCallback)
    print(f"Collision within manager 1?: {cdata.result.is_collision}")
    print("")

    assert cdata.result.is_collision is False

    ddata = fcl.DistanceData()
    manager1.distance(ddata, fcl.defaultDistanceCallback)
    print(f"Closest distance within manager 1?: {ddata.result.min_distance}")
    print("")

    assert ddata.result.min_distance > 0.0

    req = fcl.CollisionRequest(num_max_contacts=10, enable_contact=True)
    rdata = fcl.CollisionData(request=req)
    manager1.collide(o2, rdata, fcl.defaultCollisionCallback)
    print(f"Collision between manager 1 and Data?: {rdata.result.is_collision}")
    print("Contacts:")
    for c in rdata.result.contacts:
        print(f"\tO1: {c.o1}, O2: {c.o2}")
    print("")

    assert rdata.result.is_collision is False

    # -----------------------
    if save_results:
        res_dict = {
            "collision_in_manager_1": cdata.result.is_collision,
            "closest_dist_in_manager_1": ddata.result.min_distance,
            "collision_manager_1_data": rdata.result.is_collision,
        }
        results["pointcloud_with_manager"] = res_dict
        with open("fcl_result.json", "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    test_fcl(save_results=True)
