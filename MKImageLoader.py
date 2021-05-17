from PIL import Image, ImageOps
from PIL.ImageEnhance import Sharpness
import numpy as np

class MKImageLoader:
    #                                  (left, upper, right, lower)
    def __init__( self, path, main_box=(552,  51,    1228,  670)):
        im = Image.open( path)
        self.main_region = im.crop( main_box)
        self.extract_player_regions()
        self.find_user_rank()
        self.preprocess_player_regions()
        
    def extract_player_regions( self, max_players=12, player_box_height=47, player_box_gap=5):
        boxfun = lambda p: (0,
                            p*(player_box_height+player_box_gap),
                            self.main_region.size[0]-1,
                            p*(player_box_height+player_box_gap)+player_box_height)
        self.player_regions = [ self.main_region.crop( boxfun(p)) for p in range(max_players) ]
        
    def find_user_rank( self, ref_color=[239,235,20]):
        samplefun = lambda r: np.array(r)[0:9,150:,:].reshape((-1,3)).mean( axis=0)
        self.user_rank = np.argmin( [ np.linalg.norm( samplefun(r) - ref_color) for r in self.player_regions ])
        
    def preprocess_player_regions( self):
        self.player_regions = [ pr.convert(mode='L') for pr in self.player_regions ]
        self.player_regions[self.user_rank] = ImageOps.invert( self.player_regions[self.user_rank])
        self.player_regions = [ Sharpness(pr).enhance(5) for pr in self.player_regions ]
        
    def _extract( self, height, width, redges, tedge):
        boxfun = lambda e: (e-width, tedge, e, tedge+height)
        return [ [ pr.crop( boxfun(e)) for e in redges ] for pr in self.player_regions ]
        
    def _get_as_array( self, _digits):
        arrayfun = lambda x: np.expand_dims( np.array( x), axis=-1)
        catfun = lambda x: np.concatenate( x, axis=-1)
        digits = [ catfun( [arrayfun(digit) for digit in player_digits]) for player_digits in _digits ]
        return catfun( [arrayfun( digit) for digit in digits])
        
    def get_vr_digits( self):
        vr_digits = self._extract( 23, 15, [594,612,629,646,664], 15)
        return self._get_as_array( vr_digits)
        
    def get_pts_digits( self):
        pts_digits = self._extract( 18, 12, [540,553], 20)
        return self._get_as_array( pts_digits)
        
    def get_pts_signs( self):
        pts_signs = self._extract( 13, 13, [520], 22)
        return self._get_as_array( pts_signs)
        
def load_images( filepaths):
    vr_digits  = []
    pts_digits = []
    pts_signs  = []
    user_rank  = []
    for f in filepaths:
        mkil = MKImageLoader(f)
        vr_digits.append( mkil.get_vr_digits())
        pts_digits.append( mkil.get_pts_digits())
        pts_signs.append( mkil.get_pts_signs())
        user_rank.append( mkil.user_rank)
    catfun = lambda datalist: np.concatenate( [np.expand_dims(d,axis=-1) for d in datalist], axis=-1)
    return catfun(vr_digits), catfun(pts_digits), catfun(pts_signs), user_rank