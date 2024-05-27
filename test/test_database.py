import numpy as np
import peewee as pw

from Database import to_string, fr_string


def test_conversion_array_str() :
    DB = pw.SqliteDatabase(':memory:')

    class TestModel(pw.Model):

        field = pw.TextField()

        class Meta:
            database = DB

    DB.connect()
    DB.create_tables([TestModel], safe=False)

    n_tests = 100000
    test_arr = np.random.uniform(size=(n_tests, 2))

    test = TestModel.create(field=to_string(test_arr))
    test.save()

    test_retrieved = TestModel.get(TestModel.id == 1)
    test_arr_retrieved = fr_string(test_retrieved.field)
    av_diff = np.sum(np.abs(test_arr.flatten() - test_arr_retrieved))

    assert np.isclose(av_diff/n_tests, 0, atol=1e-16), f'Average difference too high: {av_diff/n_tests}'
    assert np.isclose(av_diff, 0, atol=1e-16), f'Total difference too high: {av_diff}'
