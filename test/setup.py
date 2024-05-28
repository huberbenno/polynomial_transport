import peewee as pw

from Database import *

def database() :
    DB = pw.SqliteDatabase(':memory:')
    DB.connect()
    DB.create_tables([ConvolutionDBO, GaussianDBO, GaussianMmDBO, GaussianPosteriorDBO, RosenbrockDBO, MultiIndexSetDBO,
                      MultiIndexSetAnisotropicDBO, SurrogateDBO, SurrogateEvalDBO])
