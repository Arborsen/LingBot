import random
from types import SimpleNamespace
import unittest

from scene_repair import fresh_clutter_call,capture,restore


class SceneRepairTests(unittest.TestCase):
    def test_clutter_bounds_are_repeatable_and_caller_owned_lists_unchanged(self):
        def original(env,cluttered_numbers=10,xlim=None,ylim=None,zlim=None):
            xlim[0]+=env.table_xy_bias[0];xlim[1]+=env.table_xy_bias[0]
            ylim[0]+=env.table_xy_bias[1];ylim[1]+=env.table_xy_bias[1]
            return xlim,ylim,zlim
        call=fresh_clutter_call(original);env=SimpleNamespace(table_xy_bias=[.3,.1])
        results=[call(env) for _ in range(120)]
        self.assertTrue(all(result==results[0] for result in results))
        self.assertAlmostEqual(results[0][0][0],-.29)
        x=[-1,1];y=[-2,2];z=[.7]
        call(env,xlim=x,ylim=y,zlim=z)
        self.assertEqual((x,y,z),([-1,1],[-2,2],[.7]))

    def test_native_cpu_snapshot_restores_pose_velocity_and_rng(self):
        import numpy as np
        import sapien
        import torch
        scene=sapien.Scene([sapien.physx.PhysxCpuSystem()])
        builder=scene.create_actor_builder()
        builder.add_box_collision(half_size=[.01,.01,.01]);actor=builder.build(name='test')
        actor.set_pose(sapien.Pose([.1,.2,.3]));body=actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        body.set_linear_velocity([.01,.02,.03]);body.set_angular_velocity([.1,.2,.3])
        env=SimpleNamespace(scene=scene,task_name='dump_bin_bigbin',table_z_bias=0.,table_texture=0,
            wall_texture=0,record_cluttered_objects=[])
        state=capture(env)
        expected=(random.random(),np.random.rand(),torch.rand(1).item())
        actor.set_pose(sapien.Pose([2,3,4]));body.set_linear_velocity([1,2,3])
        restored=restore(env,state)
        self.assertTrue(np.allclose(restored['actors'][0]['pose'],state['actors'][0]['pose']))
        self.assertEqual(expected,(random.random(),np.random.rand(),torch.rand(1).item()))
        changed=dict(state,configuration=dict(state['configuration'],table_texture=999))
        with self.assertRaisesRegex(AssertionError,'configuration differs'):restore(env,changed)


if __name__=='__main__':unittest.main(verbosity=2)
