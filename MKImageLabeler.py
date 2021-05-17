from MKImageLoader import load_images
import numpy as np
import pandas as pd
import PIL
from pigeon import annotate
from IPython.display import display

class MKImageLabeler:
    
    def __init__( self, filepaths):
        self.annotations = dict()
        self.filepaths = filepaths
        self.vr_digits,self.pts_digits,self.pts_signs,_ = load_images( filepaths)
        
    def _manually_label( self, data, options, max_annotations):
        s = data.shape
        indices = [ (i,j,k) for k in range(s[4]) for j in range(s[3]) for i in range(s[2]) ]
        display_fn = lambda idx: display( PIL.Image.fromarray( data[:,:,idx[0],idx[1],idx[2]]))
        n = min( s[2]*s[3]*s[4], max_annotations)
        return annotate( indices[:n], options=options, display_fn=display_fn)
        
    def label_vr_digits( self, max_annotations):
        self.annotations['vr_digits'] = self._manually_label( self.vr_digits, None, max_annotations)
        
    def label_pts_digits( self, max_annotations):
        self.annotations['pts_digits'] = self._manually_label( self.pts_digits, None, max_annotations)
        
    def label_pts_signs( self, max_annotations):
        self.annotations['pts_signs'] = self._manually_label( self.pts_signs, ['+','-','b'], max_annotations)
        
    def _get_df( self, annotations):
        df = pd.DataFrame( [ {'digit':a[0][0], 'rank':a[0][1], 'file':a[0][2], 'label':a[1]} for a in annotations ])
        df['path'] = [self.filepaths[f] for f in df['file']]
        return df[['rank','digit','path','label']]
        
    def vr_digits_as_df( self):
        return self._get_df( self.annotations['vr_digits'])
        
    def pts_digits_as_df( self):
        return self._get_df( self.annotations['pts_digits'])
        
    def pts_signs_as_df( self):
        return self._get_df( self.annotations['pts_signs'])