import collections
import copy
import json
import unittest

import numpy as np
import torch
from augmentation import (sample_augmentation,apply_augmentation,choose_occlusion,
                          foreground_mask,prepare_images_with_recipe)


class AugmentationTests(unittest.TestCase):
    def combined(self):
        while True:
            p=sample_augmentation()
            if p['branch']=='combined':return p

    def test_recipe_probabilities(self):
        torch.manual_seed(42);rows=[sample_augmentation() for _ in range(20000)]
        self.assertLess(abs(sum(p['branch']=='original' for p in rows)/len(rows)-.1),.015)
        aug=[p for p in rows if p['branch']=='combined']
        for flag,expected in [('color',.8),('affine',.5),('occlusion',.95),('gaussian_noise',.5),('salt_pepper',.15),('jpeg',.3)]:
            self.assertLess(abs(sum(p[flag] for p in aug)/len(aug)-expected),.02,flag)
        for name,expected in [('gaussian',.25),('motion',.1),('none',.65)]:
            self.assertLess(abs(sum(p['blur']==name for p in aug)/len(aug)-expected),.02)
        for p in aug:
            self.assertTrue(.25<=p['brightness']<=1.10 and .5<=p['contrast']<=1.5 and .4<=p['saturation']<=1.6)
            self.assertTrue(.9<=p['zoom']<=1.1 and abs(p['dx'])<=.06 and abs(p['dy'])<=.06)
            self.assertTrue(35<=p['jpeg_quality']<=90)

    def test_replay_shape_dtype_and_original(self):
        image=torch.randint(0,256,(3,64,80),dtype=torch.uint8)
        self.assertTrue(torch.equal(image,apply_augmentation(image,{'branch':'original'})))
        for blur in ['gaussian','motion','none']:
            p=self.combined();p.update(blur=blur,jpeg=True,occlusion=True,gaussian_noise=True,salt_pepper=True)
            a=apply_augmentation(image,p);b=apply_augmentation(image,p)
            self.assertTrue(torch.equal(a,b));self.assertEqual(a.shape,image.shape);self.assertEqual(a.dtype,image.dtype)
            repeated=apply_augmentation(torch.stack([image,image]),p)
            self.assertTrue(torch.equal(a,repeated[0]) and torch.equal(a,repeated[1]))
        f=image.float()/255;p=self.combined();a=apply_augmentation(f,p)
        self.assertEqual(a.dtype,f.dtype);self.assertTrue(0<=a.min()<=a.max()<=1)

    def test_occlusion_avoids_both_temporal_masks(self):
        a=torch.full((3,100,100),120,dtype=torch.uint8);b=a.clone();a[:,5:20,5:20]=250;b[:,5:20,75:90]=250
        p=self.combined();p['affine']=False
        box=choose_occlusion([a,b],p)
        if box is not None:
            x,y,w,h=[round(v*100) for v in box]
            for image in [a,b]:
                mask=foreground_mask(image.permute(1,2,0).numpy()/255)
                self.assertFalse(mask[y:y+h,x:x+w].any())
        # A fully protected/complex image must be permitted to skip masking.
        noisy=torch.randint(0,256,(3,100,100),dtype=torch.uint8)
        self.assertIsNone(choose_occlusion([noisy],p))

    def test_current_future_teacher_consistency(self):
        image=torch.randint(0,256,(3,48,64),dtype=torch.uint8)
        obs={'image':{'cam':image},'future_image':{'cam':image.clone()},'state':torch.zeros(14)}
        def original(image_processor,observation,image_keys=None,train=False,use_depth_align=False,
                     return_image_grid_thw=False,augment_params=None,return_augment_params=False):
            self.assertFalse(train)
            x=observation['image']['cam']
            out=(x.clone(),None,x.clone(),None)
            return out+(augment_params,) if return_augment_params else out
        now=prepare_images_with_recipe(original,None,obs,train=True,return_augment_params=True)
        future=prepare_images_with_recipe(original,None,{**obs,'image':obs['future_image']},train=True,augment_params=now[-1])
        self.assertTrue(torch.equal(now[0],now[2]))
        self.assertTrue(torch.equal(now[0],future[0]))
        self.assertTrue(torch.equal(obs['image']['cam'],image))

    def test_affine_exposed_border_remains_black(self):
        image=torch.full((3,100,100),180,dtype=torch.uint8)
        p=self.combined();p.update(affine=True,zoom=1.,dx=.06,dy=.06,color=True,
            contrast=.5,gaussian_noise=True,jpeg=True,occlusion=False)
        result=apply_augmentation(image,p)
        self.assertEqual(int(result[:,:5,:].max()),0)
        self.assertEqual(int(result[:,:,:5].max()),0)


if __name__=='__main__':unittest.main(verbosity=2)
