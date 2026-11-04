"""Bind legacy SAPIEN renderer and scene to the assigned physical GPU by PCI.

CUDA_VISIBLE_DEVICES alone is not a Vulkan device selector. Installed SAPIEN3's
legacy SapienRenderer wrapper ignores keyword device arguments, so patch only
the process-local constructors; preserve CPU PhysX and all rendering settings.
"""
import os


def normalized_pci(value):
    value=value.lower().removeprefix('pci:')
    domain,bus,device=value.split(':')
    return domain[-4:]+':'+bus+':'+device


def install(plan):
    import sapien
    from sapien.pysapien.render import SapienRenderer as NativeRenderer
    from sapien.wrapper.scene import Scene
    physical=os.environ['CUDA_VISIBLE_DEVICES']
    if physical not in ['1','2']:raise RuntimeError(f'Evaluation is restricted to GPU1/2: {physical}')
    alias=plan['render_devices'][physical]
    selected=sapien.Device(alias)
    expected=normalized_pci(alias)
    if normalized_pci(selected.pci_string)!=expected or not selected.can_render():
        raise RuntimeError(f'Render device does not match GPU{physical}: {selected}')
    original_scene=Scene.__init__
    def scene_init(self,systems=None):
        if systems is None:
            renderer=sapien.render.RenderSystem(selected)
            if normalized_pci(renderer.device.pci_string)!=expected:
                raise RuntimeError('SAPIEN selected the wrong rendering GPU')
            systems=[sapien.physx.PhysxCpuSystem(),renderer]
        original_scene(self,systems)
    def renderer_init(self,**kwargs):NativeRenderer.__init__(self,selected)
    Scene.__init__=scene_init
    sapien.SapienRenderer.__init__=renderer_init
    print(f'BOUND_SAPIEN_RENDER physical_gpu={physical} pci={expected}; CPU PhysX unchanged',flush=True)
    return selected
