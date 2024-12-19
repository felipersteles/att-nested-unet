# Nested U-net
Este repósitorio contém a implementação do código baseado no artigo <i>UNet++: A Nested U-Net Architecture for Medical Image Segmentation<i> ([doi](https://doi.org/10.48550/arXiv.1807.10165)).

Diversos experimentos foram realizados para chegar a este resultado, entretato é nítido que ainda tem muito a ser melhorado, portanto caso veja alguma falha no experimento, pode se sentir à vontade para compartilhar seu comentário na sessão de <i>issues<i> do repositório.

## Arquitetura
A arquitetura desta rede é baseada em skip connections com os nós totalmente conectados, como pode ser vista à seguir.
<div align='center'>
<image src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQoAwB8UmuAwZ3tDKFB1NA2lZVe_aIq8sHeCQ&s"/>
</div>
Um detalhe importante é que utiliza-se também um método conhecido como Supervisão profunda, onde ao invés de se avaliar somente a sáida da rede, avalia-se também a sáida dos últimos quatro nós da camada mais alta.

## Experimentos

Para realizar a busca pelo melhor parâmetro, foi-use utilizada a técnica de <i>grid search<i>, que consiste em escolher uma sequência ou sequências de hiperparâmetros e rodar o mesmo experimentos com a mesma base com a finalidade de encontrar a combinação que entregue o melhor resultado esperado. No caso deste estudo, foi-se monitarada a Loss.

## Métricas

### Dice
O dice ele avalia a segmentação com base na sobreposição entre a máscara predita e a mascara correta, sendo ideal para objetos considerados pequenos.

<div align="center">
<image width="300" src="https://miro.medium.com/v2/resize:fit:1400/1*tSqwQ9tvLmeO9raDqg3i-w.png"/>
</div>

### Indice de Jaccard (IoU)
O indice de jaccard avalia com precisão a segmentação, sendo calculado com base na máscara original e máscara encontrada pelo modelo.

<div align="center">
<image width="200" src="https://idiotdeveloper.com/wp-content/uploads/2023/01/iou-1024x781.webp"/>
</div>

## Como rodar
Aqui esta o passo-a-passo de comer executar os experimentos realizados neste estudo.

### Clonando o repositório 

```
# Clonando o repositório
$ git clone https://github.com/felipersteles/att-nested-unet.git

# Entrando no diretório
$ cd att-nested-unet

```

### Preparando os dados
Antes de realizar o pre processamento completo, aconselha-se treinar um classificador para o processo ficar mais preciso. Os códigos estão no notebool [train_classificator](./train_classificator.ipynb) 
```
# Entre no diretório de que estão os arquivos
$ cd preprocess

# Transformar 3d para 2d
$ python transform.py

# Classificar. Será gerado um arquivo em Json com os dados que não possuem e os que possuem máscara a ser segmentada.
$ python classify.py

# Filtrar os dados com o classificador treinado com base no arquivo 
$ python filter.py

# Calcular atlas probabílistico
$ python atlas.py 

# Corte da image com base no atlas
$ python crop.py

```

### Buscando os melhores hiperparâmetros

```
# Para rodar o grid search basta escolher o diretório onde estão os dados.
$ python grid_search.py --data_dir diretorio/dos/dados

```

### Treinando o modelo
```
# O treino possui muitos alguns paramêtros que podem ser configurados antes da execução
$ python train.py 
    --data_dir diretorio/dos/dados 
    --batch_size 6 
    --epochs 300 
    --save_model True 
    --model_path diretorio/do/modelo 
    --model_name nome_do_modelo.pth


```

## Resultados
Os melhores resultados obtidos após um experimento de 100 épocas utilizando o <i>Dice<i> como referência para a <i>Loss<i> foram:

<div align="center">

| Metric             | Value   |
|--------------------|---------|
| Loss              | 0.3600  |
| Dice Coefficient  | 0.6450  |
| Jaccard Index     | 0.4760  |

</div/>


## Referências
- [loss functions](https://arxiv.org/pdf/2312.05391)
- [pancreas tumor segmentation](https://doi.org/10.3389/fonc.2024.1328146)


