#!/usr/bin/python
import peewee as pw

DB = pw.SqliteDatabase(':memory:')


class BaseModel(pw.Model) :
    class Meta:
        database = DB

class Model(BaseModel) :
    mtype = pw.TextField()


class Parameter(BaseModel) :
    model = pw.ForeignKeyField(Model, backref='param')
    p_id  = pw.IntegerField()
    value = pw.TextField()


class Measurement(BaseModel) :
    param = pw.ForeignKeyField(Parameter, backref='measurement')
    m_id  = pw.IntegerField()

    class Meta:
        primary_key = pw.CompositeKey('param', 'm_id')


if __name__ == '__main__' :
    DB.connect()
    DB.create_tables([Model, Parameter, Measurement])
    model_params = {'mtype' : 'convolution'}

    model = Model.get_or_create(**model_params)[0]
    param = Parameter.get_or_create(**{'model' : model.id, 'p_id' : 1, 'value' : '1 2 3'})[0]
    #msrmt = Measurement.get_or_create(param=param.id, m_id=1, y_val='1 2 3 5')[0]


