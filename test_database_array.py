#!/usr/bin/python
import numpy as np
import peewee as pw
import io, base64

DB = pw.SqliteDatabase(':memory:')

class TestModel(pw.Model):

    field = pw.TextField()

    class Meta:
        database = DB

def to_string(arr) : return base64.binascii.b2a_base64(arr).decode("ascii")
def fr_string(stg) : return np.frombuffer(base64.binascii.a2b_base64(stg.encode("ascii")))

if __name__ == '__main__' :
    DB.connect()
    DB.create_tables([TestModel], safe=False)

    n_tests = 100000
    test_arr = np.random.uniform(size=(n_tests, 2))

    test = TestModel.create(field=to_string(test_arr))
    test.save()

    test_retrieved = TestModel.get(TestModel.id == 1)
    test_arr_retrieved = fr_string(test_retrieved.field)
    print(test_arr_retrieved)
    av_diff = np.sum(np.abs(test_arr.flatten() - test_arr_retrieved))

    print('Average difference over {} trials: {:.16f}'.format(n_tests, av_diff/n_tests))
    print('  Total difference over {} trials: {:.16f}'.format(n_tests, av_diff))
