# Tarefa 1 - Análise Exploratória de Dados

## Introdução

Nesta tarefa, explorei um conjunto de dados contendo informações sobre dois grupos de participantes, denominados Grupo A e Grupo B. O meu objetivo principal foi compreender as características dos dados, identificar padrões e verificar se existem diferenças relevantes entre os grupos em relação a duas variáveis: número de passos diários (Steps) e índice de massa corporal (BMI).

## Sobre os Dados

O conjunto de dados é composto por dois arquivos de texto formatados como CSV, cada um representando um grupo diferente de participantes. Ambos os arquivos possuem três colunas: um identificador único (ID), a quantidade de passos registrados em um período e o BMI do participante.

O Grupo A possui 865 participantes, enquanto o Grupo B conta com 921 participantes. Essa diferença no tamanho das amostras já é um ponto importante a considerar em eventuais comparações estatísticas futuras, embora não comprometa a análise exploratória realizada aqui.

## Análise Descritiva

### Grupo A

Ao calcular as estatísticas descritivas do Grupo A, observei que a média de passos diários ficou em torno de 8.997, com um desvio padrão de 3.407. Isso indica uma variabilidade considerável entre os participantes, uma vez que alguns registraram apenas 3.011 passos enquanto outros atingiram quase 15.000. A mediana de 9.075 está bem próxima da média, sugerindo uma distribuição relativamente simétrica para os passos.

Em relação ao BMI, a média foi de 26,02, com desvio padrão de 3,86. Os valores variaram de 12,8 (abaixo do peso) até 38,6 (obesidade grau I). A mediana de 25,9 praticamente coincide com a média, reforçando a simetria da distribuição.

### Grupo B

O Grupo B apresentou resultados bastante semelhantes. A média de passos foi de 9.061, praticamente igual à do Grupo A, com um desvio padrão ligeiramente maior (3.493). A faixa de valores também foi similar, variando de 3.015 a 14.993 passos.

Para o BMI, a média foi de 25,88, praticamente idêntica ao Grupo A. No entanto, o desvio padrão foi um pouco maior (4,03), e o valor máximo atingiu 42,3, um ponto que chama atenção e que discuto adiante.

## Visualizações

### Distribuição de Passos

![Steps Boxplot](steps_boxplot.png)

O boxplot de passos mostra uma semelhança notável entre os dois grupos. As medianas estão praticamente alinhadas em torno de 9.000 passos, e a dispersão é muito parecida. Os dois grupos apresentam uma cauda superior mais longa, indicando que há mais participantes com altas contagens de passos do que com valores muito baixos. Não há outliers evidentes nesta variável, o que sugere que os valores extremos estão dentro de uma faixa esperada para o tipo de atividade monitorada.

### Distribuição de BMI

![BMI Boxplot](bmi_boxplot.png)

No boxplot de BMI, as distribuições também são bastante similares entre os grupos, com medianas em torno de 25,9, valor que se encontra na faixa de "sobrepeso" segundo classificações da OMS. No entanto, o Grupo B apresenta alguns outliers na parte superior, com o valor mais extremo atingindo 42,3. Esses casos representam participantes com obesidade grau II e mereceriam uma investigação mais aprofundada para entender se são erros de medição ou participantes com condições específicas.

### Relação entre Passos e BMI

![Steps vs BMI](steps_vs_bmi_scatter.png)

O gráfico de dispersão que cruza passos com BMI é talvez a visualização mais reveladora. Nele, vejo que os pontos estão bastante dispersos, sem formar um padrão claro de correlação. Participantes com poucos passos apresentam BMIs variados, assim como aqueles com muitos passos. Essa dispersão indica que, pelo menos nestes dados, o número de passos diários por si só não parece ser um bom preditor do BMI.

Chama atenção que tanto no Grupo A quanto no Grupo B a nuvem de pontos tem uma forma bastante similar, reforçando a ideia de que os dois grupos são comparáveis em termos de comportamento e perfil.

## Observações Adicionais

Alguns pontos que julguei relevantes destacar:

- A média de BMI de ambos os grupos está ligeiramente acima de 25, o que classifica a maioria dos participantes como "sobrepeso". Isso pode ser relevante para decisões sobre programas de saúde ou intervenções.

- A variabilidade nos passos é alta em ambos os grupos. Isso pode indicar que a população é bastante heterogênea em termos de hábitos de atividade física, com desde pessoas bastante sedentárias até muito ativas.

- A presença de outliers de BMI no Grupo B, especialmente o valor de 42,3, pode distorcer médias e outras medidas de tendência central caso não sejam tratados adequadamente em análises futuras.

- A ausência de correlação visível entre passos e BMI não significa necessariamente que não existe relação entre as variáveis, pode ser que ela seja fraca, não linear, ou que outros fatores (como dieta, idade, condições de saúde) tenham papel mais relevante.

## Conclusão

De modo geral, a análise exploratória revelou que os dois grupos são bastante similares tanto em termos de atividade física quanto de composição corporal. Não há diferenças marcantes que indiquem que um grupo é significativamente diferente do outro. Os dados apresentam variabilidade considerável, o que é esperado em uma população diversa, e a relação entre passos e BMI não se mostrou forte o suficiente para drew conclusões simplistas.

Para uma análise mais aprofundada, seria recomendável aplicar testes estatísticos formais (como o teste t de Student para comparar médias ou o teste de Kolmogorov-Smirnov para comparar distribuições), além de considerar a inclusão de outras variáveis que possam explicar melhor o BMI dos participantes.
