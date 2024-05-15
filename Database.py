import base64
import numpy as np
import peewee as pw

#TODO unique constraints
DB = pw.SqliteDatabase('data.db')
# DB = pw.SqliteDatabase(':memory:')


def to_string(arr) : return base64.binascii.b2a_base64(np.array(arr)).decode("ascii")
def fr_string(stg) : return np.frombuffer(base64.binascii.a2b_base64(stg.encode("ascii")))


class BaseModel(pw.Model) :
    class Meta:
        database = DB


# ---------- Forward Models --------------------

class ConvolutionDBO(BaseModel) :
    dim   = pw.IntegerField()
    basis = pw.TextField()
    alpha = pw.IntegerField()
    nquad = pw.IntegerField()
    wkern = pw.DoubleField()


# ---------- Densities --------------------

class GaussianDBO(BaseModel) :
    dim  = pw.IntegerField()
    mean = pw.TextField()
    cova = pw.TextField()
    diag = pw.BooleanField()

    def recover_cova(self) :
        return fr_string(self.cova).reshape((self.dim, self.dim))


class GaussianMmDBO(BaseModel) :
    dim  = pw.IntegerField()
    gauss1 = pw.ForeignKeyField(GaussianDBO)
    gauss2 = pw.ForeignKeyField(GaussianDBO)
    gauss3 = pw.ForeignKeyField(GaussianDBO, null=True)
    gauss4 = pw.ForeignKeyField(GaussianDBO, null=True)
    gauss5 = pw.ForeignKeyField(GaussianDBO, null=True)


class RosenbrockDBO(BaseModel) :
    a = pw.DoubleField()
    b = pw.DoubleField()
    theta = pw.DoubleField()
    centr = pw.TextField()
    scale = pw.DoubleField()


class GaussianPosteriorDBO(BaseModel) :
    forwd = pw.ForeignKeyField(ConvolutionDBO)
    gauss = pw.ForeignKeyField(GaussianDBO)
    truep = pw.TextField()
    noise = pw.DoubleField()
    xmeas = pw.TextField()


# ---------- Indexsets --------------------

class MultiIndexSetDBO(BaseModel) :
    dim   = pw.IntegerField()
    mode  = pw.TextField()
    order = pw.IntegerField()

    size  = pw.IntegerField(null=True)
    ctime = pw.DoubleField(null=True)
    #TODO create or dump binaries?


class MultiIndexSetAnisotropicDBO(BaseModel) :
    dim    = pw.IntegerField()
    weight = pw.TextField()
    thresh = pw.DoubleField()

    size   = pw.IntegerField(null=True)
    ctime  = pw.DoubleField(null=True)


# ---------- Surrogate --------------------

class SurrogateDBO(BaseModel) :
    target    = pw.TextField()
    target_id = pw.IntegerField()
    multis    = pw.TextField()
    multis_id = pw.IntegerField()
    pmode  = pw.TextField()
    condnr = pw.DoubleField(null=True)
    closei = pw.DoubleField(null=True)

    nevals = pw.IntegerField(null=True)
    coeffs = pw.TextField(null=True)
    ctime  = pw.DoubleField(null=True)


class SurrogateEvalDBO(BaseModel):
    surrog = pw.ForeignKeyField(SurrogateDBO, backref='evals')

    # Data
    approx = pw.DoubleField(null=True)
    hedist = pw.DoubleField(null=True)
    nevals = pw.IntegerField(null=True)
    accurc = pw.DoubleField(null=True)
    ctime  = pw.DoubleField(null=True)


if __name__ == '__main__' :
    import util.log

    util.log.print_start('Setting up database...')
    DB.connect()
    DB.create_tables([ConvolutionDBO, GaussianDBO, GaussianMmDBO, GaussianPosteriorDBO, RosenbrockDBO, MultiIndexSetDBO,
                      MultiIndexSetAnisotropicDBO, SurrogateDBO, SurrogateEvalDBO])
    util.log.print_done_ctd()
