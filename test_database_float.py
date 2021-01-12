#!/usr/bin/python
import numpy as np
import peewee as pw

DB = pw.SqliteDatabase(':memory:')

class TestModel(pw.Model):

    field = pw.DoubleField()

    class Meta:
        database = DB


if __name__ == '__main__' :
    DB.connect()
    DB.create_tables([TestModel], safe=False)

    n_tests = 100000
    av_diff = 0
    for i in range(n_tests) :
        test_field = np.random.randn()
        test = TestModel.create(field=test_field)
        test.save()

        test_retrieved = TestModel.get(TestModel.id == i+1)
        av_diff += np.abs(test_field - test_retrieved.field)

    print('Average difference over {} trials: {:.16f}'.format(n_tests, av_diff/n_tests))
    print('  Total difference over {} trials: {:.16f}'.format(n_tests, av_diff))
