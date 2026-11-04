"""Pixel-only 10/90 recipe; deterministic across views, time and loader restart.

Probabilities are conditional on the 90% combined branch. Foreground protection
uses conservative image edges/contrast and a workspace guard, not a semantic
detector. Occlusion is omitted if no jointly safe current/future rectangle exists.
"""
from io import BytesIO
import inspect

import cv2
import numpy as np
from PIL import Image
import torch

cv2.setNumThreads(1)
POLICY = '17000_lr75_aug1090_task15_v2'
RECIPE = dict(original_probability=.10, combined_probability=.90,
    conditional_probabilities=dict(color=.80, affine=.50, occlusion_attempt=.95,
        gaussian_noise=.50, salt_pepper=.15, gaussian_blur=.25,
        motion_blur=.10, jpeg=.30),
    brightness=[.25,1.10], contrast=[.5,1.5], saturation=[.4,1.6],
    temperature_red_blue_gain_delta=[-.05,.05], zoom=[.90,1.10],
    translation_fraction=[-.06,.06], affine_border='black',
    occlusion_width_height_fraction=[.04,.10], occlusion_candidates=64,
    occlusion_modes=['solid','mosaic','pixel_noise'],
    foreground='union of conservative edge/contrast mask plus workspace guard for current/future; skip if unsafe',
    gaussian_noise_sigma=[1/255,3/255], salt_pepper_pixel_fraction=[.001,.005],
    gaussian_blur_kernel=3, gaussian_blur_sigma=[.1,.5],
    motion_blur_kernel=[3,5], motion_blur_orientation=['horizontal','vertical','diagonal','antidiagonal'],
    jpeg_quality=[35,90], blur_categories_mutually_exclusive=True,
    clean_only=True)


def _uniform(a,b):return a+(b-a)*torch.rand(()).item()


def sample_augmentation(reference=None):
    if torch.rand(()).item()<.1:return dict(branch='original',policy=POLICY)
    blur=torch.rand(()).item()
    return dict(branch='combined',policy=POLICY,
        color=torch.rand(()).item()<.80,
        brightness=_uniform(.25,1.10),contrast=_uniform(.5,1.5),saturation=_uniform(.4,1.6),
        temperature=_uniform(-.05,.05),
        affine=torch.rand(()).item()<.50,zoom=_uniform(.9,1.1),
        dx=_uniform(-.06,.06),dy=_uniform(-.06,.06),
        occlusion=torch.rand(()).item()<.95,
        occlusion_width=_uniform(.04,.10),occlusion_height=_uniform(.04,.10),
        occlusion_mode=['solid','mosaic','pixel_noise'][torch.randint(3,()).item()],
        gaussian_noise=torch.rand(()).item()<.50,noise_sigma=_uniform(1/255,3/255),
        salt_pepper=torch.rand(()).item()<.15,salt_pepper_fraction=_uniform(.001,.005),
        blur='gaussian' if blur<.25 else 'motion' if blur<.35 else 'none',
        blur_sigma=_uniform(.1,.5),motion_size=3+2*torch.randint(2,()).item(),
        motion_direction=torch.randint(4,()).item(),
        jpeg=torch.rand(()).item()<.30,jpeg_quality=torch.randint(35,91,()).item(),
        seed=torch.randint(0,2**31,()).item(),occlusion_boxes={})


def _array(visual):
    if visual.ndim!=3 or visual.shape[0]!=3:raise ValueError('Expected RGB C,H,W')
    x=visual.detach().cpu().numpy().transpose(1,2,0)
    scale=255. if visual.dtype==torch.uint8 or float(x.max())>1. else 1.
    return np.clip(x.astype(np.float32)/scale,0,1),scale


def _geometry(x,p):
    if not p.get('affine'):return x.copy()
    h,w=x.shape[:2];z=p['zoom']
    matrix=np.array([[z,0,(1-z)*(w-1)/2+p['dx']*w],
                     [0,z,(1-z)*(h-1)/2+p['dy']*h]],dtype=np.float32)
    return cv2.warpAffine(x,matrix,(w,h),flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))


def foreground_mask(image):
    """Conservative heuristic: preserve contours, texture and central workspace."""
    u=np.round(np.clip(image,0,1)*255).astype(np.uint8)
    gray=cv2.cvtColor(u,cv2.COLOR_RGB2GRAY)
    edges=cv2.Canny(gray,35,90)
    edges=cv2.dilate(edges,np.ones((5,5),np.uint8))
    contours,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    mask=edges.copy()
    cv2.drawContours(mask,contours,-1,255,thickness=cv2.FILLED)
    # Retain low-contrast features missed by the edge detector.
    local=cv2.absdiff(u,cv2.GaussianBlur(u,(0,0),2)).max(axis=2)>12
    mask[local]=255
    mask=cv2.dilate(mask,np.ones((7,7),np.uint8))
    h,w=mask.shape
    mask[int(.25*h):int(.95*h),int(.15*w):int(.85*w)]=255
    return mask.astype(bool)


def choose_occlusion(images,params):
    """One rectangle per camera, clear of the union of current/future masks."""
    arrays=[_geometry(_array(x)[0],params) for x in images]
    h,w=arrays[0].shape[:2]
    if any(x.shape[:2]!=(h,w) for x in arrays):return None
    union=np.logical_or.reduce([foreground_mask(x) for x in arrays])
    rng=np.random.default_rng(params['seed'])
    rw=max(1,round(w*params['occlusion_width']));rh=max(1,round(h*params['occlusion_height']))
    for _ in range(64):
        x=int(rng.integers(0,w-rw+1));y=int(rng.integers(0,h-rh+1))
        if not union[y:y+rh,x:x+rw].any():return [x/w,y/h,rw/w,rh/h]
    return None


def apply_augmentation(visual,params,camera_key=None):
    if params['branch']=='original':return visual
    if visual.ndim==4:
        return torch.stack([apply_augmentation(x,params,camera_key) for x in visual])
    x,scale=_array(visual);x=_geometry(x,params)
    rng=np.random.default_rng(params['seed']);h,w=x.shape[:2]
    if params['color']:
        x=x*params['brightness']
        gray=np.sum(x*np.array([.2989,.5870,.1140],np.float32),axis=2,keepdims=True)
        x=gray+(x-gray)*params['saturation']
        mean=float(np.mean(gray));x=mean+(x-mean)*params['contrast']
        x=x*np.array([1+params['temperature'],1.,1-params['temperature']],np.float32)
        x=np.clip(x,0,1)
    if params['occlusion']:
        boxes=params.get('occlusion_boxes',{})
        box=boxes.get(camera_key) if camera_key in boxes else choose_occlusion([visual],params)
        if box is not None:
            xx,yy,rw,rh=[round(a*b) for a,b in zip(box,[w,h,w,h])]
            patch=x[yy:yy+rh,xx:xx+rw]
            if params['occlusion_mode']=='solid':patch[:]=rng.random(3)
            elif params['occlusion_mode']=='pixel_noise':patch[:]=rng.random(patch.shape)
            else:
                small=cv2.resize(patch,(max(1,rw//6),max(1,rh//6)),interpolation=cv2.INTER_AREA)
                patch[:]=cv2.resize(small,(rw,rh),interpolation=cv2.INTER_NEAREST)
    if params['gaussian_noise']:x=x+rng.normal(0,params['noise_sigma'],x.shape).astype(np.float32)
    if params['salt_pepper']:
        a=rng.random((h,w));q=params['salt_pepper_fraction']/2
        x[a<q]=0;x[a>1-q]=1
    if params['blur']=='gaussian':x=cv2.GaussianBlur(x,(3,3),params['blur_sigma'])
    elif params['blur']=='motion':
        n=params['motion_size'];kernel=np.zeros((n,n),np.float32)
        direction=params['motion_direction']
        if direction==0:kernel[n//2,:]=1
        elif direction==1:kernel[:,n//2]=1
        elif direction==2:np.fill_diagonal(kernel,1)
        else:np.fill_diagonal(np.fliplr(kernel),1)
        x=cv2.filter2D(x,-1,kernel/n,borderType=cv2.BORDER_REFLECT_101)
    x=np.clip(x,0,1)
    if params['jpeg']:
        buffer=BytesIO()
        Image.fromarray(np.round(x*255).astype(np.uint8)).save(buffer,format='JPEG',quality=params['jpeg_quality'])
        buffer.seek(0);x=np.asarray(Image.open(buffer).convert('RGB'),np.float32)/255
    if params['affine']:
        # Keep exposed borders black even after photometric/noise/JPEG operations.
        valid=_geometry(np.ones((h,w,3),np.float32),params).min(axis=2)>=.999
        x[~valid]=0
    x=np.ascontiguousarray((x*scale).transpose(2,0,1))
    out=torch.from_numpy(x)
    if visual.dtype==torch.uint8:out=out.round()
    return out.to(dtype=visual.dtype,device=visual.device)


def prepare_images_with_recipe(original,*args,**kwargs):
    bound=inspect.signature(original).bind(*args,**kwargs);bound.apply_defaults()
    values=dict(bound.arguments)
    if not values['train']:return original(*args,**kwargs)
    obs=values['observation'];params=values['augment_params']
    if params is None:
        params=sample_augmentation(next(iter(obs['image'].values())))
        if params['branch']!='original' and params['occlusion']:
            for key,img in obs['image'].items():
                pair=[img]
                if key in obs.get('future_image',{}):pair.append(obs['future_image'][key])
                params['occlusion_boxes'][key]=choose_occlusion(pair,params)
    images={key:apply_augmentation(img,params,key) for key,img in obs['image'].items()}
    values['observation']={**obs,'image':images}
    # Feed identical augmented RGB to the policy, depth teacher and video teacher.
    values['train']=False;values['augment_params']=params
    return original(**values)


def install():
    from lingbotvla.data.vla_data import transform,utils
    original=transform.prepare_images
    def wrapped(*args,**kwargs):return prepare_images_with_recipe(original,*args,**kwargs)
    transform.prepare_images=wrapped;utils.prepare_images=wrapped
    transform.sample_visual_augmentation_params=sample_augmentation
    transform.apply_visual_augmentation=apply_augmentation
