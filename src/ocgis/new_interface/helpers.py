import os


def write_fiona_htmp(obj, name):
    path = os.path.join('/home/benkoziol/htmp/ocgis', '{}.shp'.format(name))
    obj.write_fiona(path)
