rm data.db

python Database.py  || exit

python Forward.py  || exit

python Densities.py  || exit

python MultiIndex.py || exit

python Surrogates.py || exit

python Transport.py || exit
