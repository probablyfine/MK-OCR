from MKImageLoader import load_images
from MKImageClassifier import MKImageClassifier
import pandas as pd
import numpy as np

class MKDataCompiler:
    
    def __init__( self, paths_to_labeled, n_splits=10, n_jobs=2):
        self.paths_to_labeled = paths_to_labeled
        self.n_splits = n_splits
        self.n_jobs = n_jobs
        self._fit()
        
    def _fit( self):
        self.vr_digits_clf  = self._get_clf( self.paths_to_labeled['vr_digits'], 'vr_digits')
        self.pts_digits_clf = self._get_clf( self.paths_to_labeled['pts_digits'], 'pts_digits')
        self.pts_signs_clf  = self._get_clf( self.paths_to_labeled['pts_signs'], 'pts_signs')
        
    def _predict( self, paths_to_images):
        vr_digits, pts_digits, pts_signs, user_ranks = load_images( paths_to_images)
        self.vr_digits_hat  = self.vr_digits_clf.predict( vr_digits)
        self.pts_digits_hat = self.pts_digits_clf.predict( pts_digits)
        self.pts_signs_hat  = self.pts_signs_clf.predict( pts_signs)
        self.user_ranks = user_ranks
        
    def _get_clf( self, path_to_labeled, kind):
        clf = MKImageClassifier( path_to_labeled, kind, n_splits=self.n_splits)
        _ = clf.tune( n_jobs=self.n_jobs)
        return clf
        
    def _y_hat_to_df( self, y_hat, label):
        s = y_hat.shape
        df = pd.DataFrame( [{ 'rank':i, 'race':j, label:y_hat.sum(axis=0)[i,j] } for j in range(s[2]) for i in range(s[1])])
        df['race'] = df['race'] + 1
        df['rank'] = df['rank'] + 1
        return df.set_index( ['race','rank'])
        
    def _clean_df( self, df):
        blankfun = lambda s,i: s[i]=='b'
        mask1 = df['VR'].apply( lambda x: blankfun(x,-1))
        mask2 = df['points'].apply( lambda x: blankfun(x,0))
        mask3 = df['points'].apply( lambda x: blankfun(x,1))
        return df[~(mask1|mask2|mask3)].applymap( lambda x: int( x.replace('b','')))
        
    def _append_user_ranks( self, df):
        df.loc[[(i+1,rank+1) for i,rank in enumerate( self.user_ranks)],'is user'] = 'x'
        return df
        
    def compile( self, paths_to_images):
        self._predict( paths_to_images)
        df_vr  = self._y_hat_to_df( self.vr_digits_hat, 'VR')
        df_pts = self._y_hat_to_df( np.concatenate( (self.pts_signs_hat,self.pts_digits_hat), axis=0), 'points')
        df_vr['points'] = df_pts['points']
        return self._append_user_ranks( self._clean_df( df_vr))
        
def append_room_stats( df):
    for race in df.index.get_level_values('race').unique():
        df.loc[(race,),'room mean']   = df.loc[(race,),'VR'].mean()
        df.loc[(race,),'room median'] = df.loc[(race,),'VR'].median()
        df.loc[(race,),'room min']    = df.loc[(race,),'VR'].min()
        df.loc[(race,),'room max']    = df.loc[(race,),'VR'].max()
        df.loc[(race,),'room std']    = df.loc[(race,),'VR'].std()
        df.loc[(race,),'norm rank']   = (df.loc[(race,)].index / df.loc[(race,)].index.max())*12
        df.loc[(race,),'nplayers']    = df.loc[(race,)].index.max()
        df.loc[(race,),'norm VR']     = (df.loc[(race,),'VR'] / df.loc[(race,),'VR'].sum()).values
    return df