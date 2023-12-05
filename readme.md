# Remoção de ruídos em audios
Este repositório contem o código para remoção de ruídos em audios. Projeto final da disciplina INF721 - Aprendizado em Redes Neurais Profundas.

## Download do Modelo:
....  // Será atualizado em breve


## Usando o modelo treinado:
Altere o `file` no arquivo `workspace/inference.py`. Após isso, rode o script:
```
cd workspace
python inference.py
```

## Treinando o modelo:
Para treinar o modelo, utilizando o mesmo dataset. Atualize os submodules do git, caso não tenha sido baixado, rode:
```
git submodule update --init --recursive
```
Após utilize o make para gerar os dados de treinamento:
```
make
```
Para treinar o modelo, utilize o script `train.py` dentro do workspace:
```
cd workspace
python train.py
```

Para treinar um modelo utilizando outro dataset basta alterar o dataset dentro da função `load_dataset` no arquivo `workspace/train.py`.
