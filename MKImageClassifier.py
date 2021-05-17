from MKImageLoader import MKImageLoader
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed

class MKImageClassifier:
    
    def __init__( self, path_to_labeled, kind, n_splits=10):
        self.kind = kind
        self.df_labeled = pd.read_excel( path_to_labeled)
        self.X = self._load_X_from_images()
        self.y = self.df_labeled['label'].astype(str).values
        self.n_splits = n_splits
        
    def _load_X_from_images( self):
        paths = self.df_labeled['path'].unique()
        if self.kind == 'vr_digits':
            image_dict = {p: MKImageLoader(p).get_vr_digits() for p in paths}
        elif self.kind == 'pts_digits':
            image_dict = {p: MKImageLoader(p).get_pts_digits() for p in paths}
        elif self.kind == 'pts_signs':
            image_dict = {p: MKImageLoader(p).get_pts_signs() for p in paths}
        else:
            raise Exception( 'kind should be vr_digits, pts_digits, or pts_signs')
        reshape_fun = lambda x: x.reshape( (-1,x.shape[2],x.shape[3]))
        image_dict = {k: reshape_fun(v) for k,v in image_dict.items()}
        idx_fun = lambda row: np.expand_dims( image_dict[row['path']][:,row['digit'],row['rank']], 0)
        return np.concatenate( [idx_fun(row) for _,row in self.df_labeled.iterrows()], axis=0).astype(float)
        
    def _xvalidate( self, ncomp, C, return_data=False):
        print( '.', end='', flush=True)
        kf = KFold( n_splits=self.n_splits, shuffle=True, random_state=0)
        y_hat = np.empty_like( self.y)
        for train_index,test_index in kf.split( self.X):
            mdl = MKImageModel( self.X[train_index], self.y[train_index], ncomp, C)
            mdl.fit()
            y_hat[test_index] = mdl.predict( self.X[test_index])
        results_dict = { 'ncomp':ncomp, 'C':C, 'accuracy':sum(y_hat==self.y)/len(self.y) }
        if return_data:
            results_dict['y_hat'] = y_hat
            results_dict['y'] = self.y.copy()
        return results_dict
        
    def _fit( self):
        self.mdl = MKImageModel( self.X, self.y, self.ncomp, self.C)
        self.mdl.fit()
        
    def tune( self, ncomps=[4,5,10,15,20,25], Cs=[0.01,0.1,0.5,1], n_jobs=2):
        print( 'tuning...', end='', flush=True)
        gs = Parallel( n_jobs=n_jobs)(delayed(self._xvalidate)(ncomp,C) for C in Cs for ncomp in ncomps)
        gs = pd.DataFrame( gs)
        gs.sort_values( by=['accuracy','ncomp','C'], ascending=[False,True,True], inplace=True)
        gs.reset_index( inplace=True, drop=True)
        self.ncomp = gs.at[0,'ncomp']
        self.C = gs.at[0,'C']
        self.accuracy = gs.at[0,'accuracy']
        self._fit()
        print( 'done.')
        print( f'classification accuracy w/ {self.n_splits}-fold xval: {self.accuracy*100}% using ncomp={self.ncomp} and C={self.C}')
        return gs
        
    def predict( self, loaded_images):
        s = loaded_images.shape
        X = loaded_images.reshape( (s[0]*s[1],-1)).T
        y_hat = self.mdl.predict( X)
        return y_hat.reshape( (s[2],s[3],s[4]))
        
eps = 1e-4
class MKImageModel:
    
    def __init__( self, X_train, y_train, ncomp, C):
        self.X_train = X_train
        self.y_train = y_train
        self.ncomp = ncomp
        self.C = C
    
    def _norm_samples( self, a):
        a = a - np.expand_dims( a.min(axis=1), axis=-1)
        return a / np.expand_dims( a.max(axis=1)+eps, axis=-1)
        
    def _norm_features( self, a):
        self.mean = np.expand_dims( a.mean(axis=0), axis=0)
        a = a - self.mean
        self.std = np.expand_dims( a.std(axis=0), axis=0)
        return a / (self.std+eps)
        
    def fit( self):
        X = self._norm_samples( self.X_train.copy())
        X = self._norm_features( X)
        self.fa = FactorAnalysis( n_components=self.ncomp)
        self.scores = self.fa.fit_transform( X)
        self.svc = LinearSVC( class_weight='balanced', dual=False, C=self.C, max_iter=10000, tol=1e-5)
        self.svc.fit( self.scores, self.y_train)
        
    def predict( self, X_test):
        X = self._norm_samples( X_test.copy())
        X = X - self.mean
        X = X / (self.std+eps)
        scores = self.fa.transform( X)
        return self.svc.predict( scores)