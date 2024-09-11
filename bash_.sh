echo "Resultados Finales
" > resultados_fin.txt


python main.py --epochs 60 --early_stop=8 --patience 3 --lr 0.0001 --wd 0.0001 --hidden_size 80 --lstm_layers 3 --Sc_norm True --seed 1 --train False
python main.py --epochs 60 --early_stop=8 --patience 3 --lr 0.0001 --wd 0.0001 --hidden_size 80 --lstm_layers 3 --Sc_norm True --seed 123 --train False
python main.py --epochs 60 --early_stop=8 --patience 3 --lr 0.0001 --wd 0.0001 --hidden_size 80 --lstm_layers 3 --Sc_norm True --seed 12 --train False

echo "
Modelo clasificación
" >> resultados_fin.txt

python main.py --epochs 60 --early_stop=8 --patience 3 --lr 0.0001 --wd 0.0001 --hidden_size 80 --lstm_layers 3 --Sc_norm True --seed 1 --clasificacion True --train False
python main.py --epochs 60 --early_stop=8 --patience 3 --lr 0.0001 --wd 0.0001 --hidden_size 80 --lstm_layers 3 --Sc_norm True --seed 123 --clasificacion True --train False
python main.py --epochs 60 --early_stop=8 --patience 3 --lr 0.0001 --wd 0.0001 --hidden_size 80 --lstm_layers 3 --Sc_norm True --seed 12 --clasificacion True --train False

echo "
Para 5 semillas en vez de 3
" >> resultados_fin.txt

python main.py --epochs 60 --early_stop=8 --patience 3 --lr 0.0001 --wd 0.0001 --hidden_size 80 --lstm_layers 3 --Sc_norm True --seed 1234 --train False
python main.py --epochs 60 --early_stop=8 --patience 3 --lr 0.0001 --wd 0.0001 --hidden_size 80 --lstm_layers 3 --Sc_norm True --seed 12345 --train False

echo "
Modelo clasificación train dev y val 2021
" >> resultados_fin.txt

python main.py --epochs 60 --early_stop=8 --patience 3 --lr 0.0001 --wd 0.0001 --hidden_size 80 --lstm_layers 3 --Sc_norm True --seed 1 --clasificacion True --train_2019 False --train False
python main.py --epochs 60 --early_stop=8 --patience 3 --lr 0.0001 --wd 0.0001 --hidden_size 80 --lstm_layers 3 --Sc_norm True --seed 12 --clasificacion True --train_2019 False --train False
python main.py --epochs 60 --early_stop=8 --patience 3 --lr 0.0001 --wd 0.0001 --hidden_size 80 --lstm_layers 3 --Sc_norm True --seed 123 --clasificacion True --train_2019 False --train False
python main.py --epochs 60 --early_stop=8 --patience 3 --lr 0.0001 --wd 0.0001 --hidden_size 80 --lstm_layers 3 --Sc_norm True --seed 1234 --clasificacion True --train_2019 False
python main.py --epochs 60 --early_stop=8 --patience 3 --lr 0.0001 --wd 0.0001 --hidden_size 80 --lstm_layers 3 --Sc_norm True --seed 12345 --clasificacion True --train_2019 False


echo "
Modelo clasificación
" >> resultados_fin.txt

python main.py --epochs 60 --early_stop=8 --patience 3 --lr 0.0001 --wd 0.0001 --hidden_size 80 --lstm_layers 3 --Sc_norm True --seed 1234 --clasificacion True 
python main.py --epochs 60 --early_stop=8 --patience 3 --lr 0.0001 --wd 0.0001 --hidden_size 80 --lstm_layers 3 --Sc_norm True --seed 12345 --clasificacion True 


