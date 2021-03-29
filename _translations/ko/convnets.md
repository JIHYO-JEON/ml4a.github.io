---
layout: chapter
title: "합성곱 신경망"
includes: [mathjax, jquery, convnetjs, dataset, convnet, visualizer]
header_image: "/images/headers/Convet_activations_kitty.jpg"
header_text: "학습된 합성곱 신경망에서 매핑된 활성화 맵, <a href=\"/guides/ConvnetViewer/\">ConvnetViewer</a> 를 통해 본<a href=\"https://github.com/ml4a/ml4a-ofx\">openframeworks app</a> 컬렉션."

---

[English](/ml4a/convnets/) [日本語](/ml4a/jp/convnets/) 

CNN 또는 합성곱 신경망(convolutional neural network)는 최근 몇 년 동안 연구에서 가장 두드러진 [신경망](/ml4a/neural_network/) 변형으로 부상하면서 딥 러닝의 핵심으로 떠올랐습니다. 이는 컴퓨터 비전의 혁신으로, 다양한 기본적인 문제들에서 최첨단 결과를 보여줬고, 자연어 처리, 컴퓨터 오디션, 강화 학습, 그리고 많은 다른 분야에서 큰 발전을 이뤘습니다. 합성곱 신경망은 오늘날 수 많은 기술 회사들의 새로운 서비스와 기능에서 쉽게 볼 수 있습니다. 다음과 같은 다양한 애플리케이션이 있습니다.

- 이미지에서 개체, 위치 및 사람을 감지하고 레이블 지정하기
- 음성을 텍스트로 변환하고 자연음의 오디오 합성하기
- 이미지와 동영상을 자연어로 설명하기
- 자율 주행 차량의 도로 추적 및 장애물을 탐색하기
- 비디오 게임 화면을 분석하여 자율 에이전트가 게임을 플레이하도록 안내하기
- 생성 모델에서 "환각과 같은" 이미지, 소리 및 텍스트를 생성하기

[최소한 현재 형태](https://plus.google.com/100849856540000067209/posts/9BDtGwCDL7D)의 합성곱 신경망은 [초기 뇌과학 연구](https://en.wikipedia.org/wiki/Hebbian_theory)에 뿌리를 두고 있습니다. 그들은 1980년 부터 존재했지만, 여러 과학 연구 커뮤니티에서 인정받게 된 것은 여러 분야의 중요한 과제에서  탁월한 성과를 거둔 최근의 일입니다. 기존 신경망에 새로운 종류의 계층을 도입함으로써 위치나 크기, 시점에 대응하는 신경망의 능력을 향상시켰습니다. 신경망은 수십 또는 수백 개의 층을 거쳐 점점 더 깊어졌고, 게임 보드 및 기타 공간 데이터 구조뿐만 아니라 이미지, 소리 심지어 계층적 구조 모델까지 되었습니다.

시각(Vision)기반 작업에서 성공했기 때문에, 창조적인 기술자들과 상호작용 디자이너이 신경망을 사용하기 시작했습니다. 움직임을 감지하는 것 뿐만 아니라, 물리적인 공간에서 물체를 적극적으로 식별하고, 묘사하고, 추적할 수 있게 합니다. [딥드림(Deepdream)](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)과 [스타일 변환(style transfer)](https://arxiv.org/abs/1508.06576)이 뉴미디어 아티스트들의 눈길을 끌었습니다.

다음 몇 장은 합성곱을 공식화하고 어떻게 동작하는지, 합성곱 신경망과 그 응용 프로그램을 다룹니다.  [다음 장](/ml4a/visualizing_convnets)에서는 그 속성을 설명하는 및 창작 및 예술적 응용 사례를 다룰 것입니다.

## 일반 신경망의 약점

합성곱 신경망이 제공하는 혁신을 이해하기 위해, 이전 장 [신경망의 내부](/ml4a/ko/looking_inside_neural_nets/)에서 자세히 다뤘던 일반 신경망의 단점을 먼저 알아보겠습니다.

학습된 단층 일반 신경망에서 입력 픽셀과 출력 뉴런 사이의 가중치는 결국 각 출력 클래스의 템플릿처럼 보인다는 것을 기억하십시오. 이는 각 클래스에 대한 모든 정보를 하나의 레이어로 파악하도록 제한되기 때문입니다. 각 템플릿은 해당 클래스에 속하는 이미지의 평균을 취한 것 같습니다.

{% include figure_multi.md path1="/images/figures/mnist_cifar_weights_review.png" caption1="MNIST 숫자 데이터로 학습된 단층 신경망의 가중치는 각 클래스에 대한 \\"템플릿\\"를 캡처합니다. 그러나 CIFAR-10과 같이 더 복잡한 클래스의 경우 클래스 내 편차가 너무 커 특징을 파악할 수 없습니다." %}

MNIST 데이터 세트의 경우, 우리는 템플릿이 상대적으로 식별 가능하여 효과적이지만 CIFAR-10의 경우 인식하기가 훨씬 더 어렵다는 것을 알 수 있습니다. 그 이유는 CIFAR-10는 MNIST보다 훨씬 많은 편차를 보이기 때문입니다. 개의 이미지는 몸을 웅크리거나 뻗고, 털 색깔이 다르고, 다른 물건들로 어수선하고, 다양한 다른 왜곡된 개를 포함할 수 있습니다. 이러한 모든 변화를 한 계층에서 학습하도록 강요된 우리의 네트워크는 모든 개 사진에 대해 매우 약한 평균을 형성할 수 있으며, 이는 일관성 있게 보이지 않는 것을 정확하게 인식하기 어렵습니다.

숨겨진 계층을 생성하여 네트워크에서 발견된 기능의 계층을 구성할 수 있는 용량을 제공함으로써 이러한 문제를 해결할 수 있습니다. 예를 들어, 우리가 MNIST를 분류하기 위해 2층 신경망을 만든다고 가정해 보죠. MNIST는 10개의 뉴런을 포함하는 숨겨진 층과 마지막 출력 층도 (이전처럼) 우리 자릿수에 대한 10개의 뉴런을 포함하고 있습니다. [네트워크를 훈련하고 가중치를 추출합니다](/demos/f_mnist_weights/). 다음 그림에서, 우리는 이전과 같은 방법을 사용하여 첫 번째 레이어 가중치를 시각화하고, 또한 막대그래프로서 숨겨진 10개의 뉴런을 10개의 출력 뉴런에 연결하는 두 번째 레이어 가중치를 시각화합니다.

{% include figure_multi.md path1="/images/figures/mnist2-combinations.jpg" caption1="MNIST에서 훈련된 2층 신경망을 위한 첫 번째 두 번째 출력 뉴런에 대한 첫 번째 레이어 가중치(위 행)와 두 번째 레이어 가중치(바 그래프)입니다. 이 그림은 <a href=\"/demos/f_mnist_weights/\">이 데모</a>에서 다시 만들 수 있습니다." %}

첫 번째 레이어 가중치는 여전히 이전과 같은 방식으로 시각화할 수 있지만, 숫자 자체와 같은 모습을 보이는 대신, 조각으로 보이거나, 아마도 모든 경우에서 다양하게 발견되는 보다 일반적인 모양과 패턴으로 보입니다. 첫 번째 행 막대 그래프는 숫자 0을 식별 출력 뉴런에 대한 각각의 숨겨진 레이어 뉴런이 얼마나 영향을 주는지를 보여줍니다. 그것은 바깥 고리를 가진 1층 뉴런들을 선호하는 것으로 보입니다. 그리고 그것은 중간에 높은 무게를 가진 뉴런들을 싫어합니다. 두 번째 행은 출력 1-뉴런에 대한 동일한 시각화로, 중간 픽셀이 높은 이미지에 대해 강한 활동을 보이는 숨겨진 뉴런을 선호합니다. 따라서 우리는 네트워크가 첫 번째 계층에서 손으로 쓴 숫자에 더 일반적인 기능을 배울 수 있고 다른 계층에는 없는 일부 자릿수에 존재할 수 있는 특징을 배울 수 있다는 것을 알 수 있습니다. 예를 들어, 외부 루프 또는 링은 8과 0에는 유용하지만 1 또는 7에는 유용하지 않습니다. 가운데를 통과하는 대각선 스트로크는 7과 2에는 유용하지만 5 또는 0에는 유용하지 않습니다. 오른쪽 상단에서 빠른 변곡은 2, 7, 9에는 유용하지만 5, 6에는 유용하지 않습니다.

CIFER-10에 관련된 예제가 있습니다. 말의 이미지의 대부분은 왼쪽과 오른쪽의 말을들을 바탕으로 템플릿을 만들 때 멍한 2 개의 목이있는 말처럼되어 버립니다. 숨겨진 레이어가있는 경우, 네트워크는 "왼쪽 말"과 "오른쪽 말"템플릿을 숨겨진 레이어에서 배우고 출력 뉴런은 각각에 대해 강한 가중치를 할당 할 수 있습니다. 이는 특별히 개선된 것은 아니지만, 규모를 확대하면서 이러한 전략이 어떻게 네트워크에 더 많은 유연성을 제공하는지 알게 되었습니다. 초기 계층에서는 보다 국소적이고 일반적으로 적용 가능한 기능을 학습할 수 있으며, 이후 계층에서 결합할 수 있습니다.

이러한 개선에도 불구하고, 네트워크가 다양한 이미지의 데이터 세트를 완전히 특성화할 수 있는 거의 끝없는 가중치 숫자의 집합을 기억하는 것은 여전히 비현실적입니다. 그 많은 정보를 포착하기 위해서는 우리가 실질적으로 저장하거나 훈련시킬 수 있는 것에 너무 많은 뉴런이 필요합니다. 합성곱 신경망의 장점은 이런 순열을 보다 효율적으로 파악할 수 있게 해준다는 것입니다.

## 구성성

우리는 어떻게 많은 종류의 이미지를 효율적으로 네트워크를 이용하여 표현할 수 있을까요? 예시를 생각해서 이 질문에 대한 직관을 얻을 수 있을 것입니다.
 
한 번도 본 적이 없는 자동차 사진을 보여드리면요. 그것이 자동차의 다양한 특징을 줄줄이 가지고 있다는 것을 관찰함으로써 자동차로 식별할 수 있을 것입니다. 즉, 앞 유리, 바퀴, 문, 배기관 등 대부분의 자동차를 구성하는 부품의 조합을 포함하고 있습니다. 이 사진과 동일한 부품의 조합에 만난 적이 없음에도 불구하고 각각의 작은 부분을 인식하고 서로 더하는 것으로 당신은 이것이 자동차의 사진임을 알게됩니다.

합성곱 신경망은 이와 비슷한 것을 시도합니다. 물체의 개별 부분을 학습하여 각각의 뉴런에 저장하고, 그것들을 더해서 더 큰 물체를 인식합니다. 이 방법에는 두 가지 장점이 있습니다. 하나는 더 적은 수의 뉴런 안에서 더 다양한 물체를 파악할 수 있다는 것입니다. 예를 들어, 10개의 바퀴 템플릿, 10개의 문 템플릿, 10개의 앞 유리 템플릿을 사용했다고 가정합니다. 따라서 우리는 30개의 템플릿 을 사용해서 $10 * 10 = 1000$ 종류의 다른 자동차를 파악할 수 있습니다. 이는 자동차가 중복되는 부분이 많은 약 1000개의 별도 템플릿을 보관하는 것보다 훨씬 효율적이다. 또한, 이러한 작은 템플릿은 다른 것의 클래스에 다시 사용할 수 있습니다. 승합차에도 휠이 있습니다. 집에도 문이 있습니다. 배에도 앞 유리가 있습니다. 더 많은 것의 클래스의 집합을 작은 부품의 다양한 조합으로 구축 할 수있어 게다가 매우 효율적으로 할 수 있습니다.

# 합성곱 신경망의 역사와 선행 

합성곱 신경망이 이러한 유형의 특징을 검출하는 방법을 단계별로 살펴보기 전에, 여기까지 설명한 문제에 대해 회선 신경망이 어떻게 진화 해 왔는지 이해하기 위해 선행하는 중요한 연구를 소개하겠습니다.

## 허블(Hubel)과 위젤(Wiesel)의 실험 (1960년대)

1960년대, 신경생리학자 [데이비드 허블](https://en.wikipedia.org/wiki/David_H._Hubel)과 [토스턴 위젤](https://en.wikipedia.org/wiki/Torsten_Wiesel)이 동물들의 시각적 피질의 특성을 조사하기 위해 일련의 실험을 수행했습니다. [가장 주목할 만한 실험들 중 하나](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1359523/)에서 그들은 TV 화면의 간단한 패턴으로 고양이의 뇌를 자극하면서 고양이의 뇌로부터의 전기적 반응을 측정했습니다. 그들이 발견한 것은 초기 시각피질에서 뉴런이 계층적 방식으로 조직되어 있다는 것이었습니다. 여기서 고양이의 망막에 연결된 첫 번째 세포는 가장자리와 막대 같은 단순한 패턴을 감지하고, 그 다음에는 초기 뉴런 활동을 결합함으로써 더 복잡한 패턴에 반응한다는 것을 알아냈습니다.

{% include figure_multi.md path1="/images/figures/hubel-wiesel.jpg" caption1="허블(Hubel)과 위젤(Wiesel)의 실험" %}

Later experiments on [macaque monkeys](http://www.cns.nyu.edu/~tony/vns/readings/hubel-wiesel-1977.pdf) revealed similar structures, and continued to refine an emerging understanding of mammalian visual processing. Their experiments would provide an early inspiration to artificial intelligence researchers seeking to construct well-defined computational frameworks for computer vision.

{% include further_reading.md title="Receptive fields, binocular interaction and functional architecture in the cat's visual cortex" author="D. H. Hubel and T. N. Wiesel" link="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1359523/" %} 


## Fukushima's Neocognitron (1982)

Hubel and Wiesel's experiments were directly cited as inspiration by [Kunihiko Fukushima](http://personalpage.flsi.or.jp/fukushima/index-e.html) in devising the [Neocognitron](http://www.cs.princeton.edu/courses/archive/spr08/cos598B/Readings/Fukushima1980.pdf), a neural network  which attempted to mimic these hierarchical and compositional properties of the visual cortex. The neocognitron was the first neural network architecture to use hierarchical layers where each layer is responsible for detecting a pattern from the output of the previous one, using a sliding filter to locate it anywhere in the image.

{% include figure_multi.md path1="/images/figures/neocognitron.jpg" caption1="Neocognitron" %}

Although the neocognitron achieved some success in pattern recognition tasks and introduced convolutional filters to neural networks, it was limited by its lack of a training algorithm to learn the filters. This meant that the pattern detectors were manually engineered for the specific task, using a variety of heuristics and techniques from computer vision. At the time, [backpropagation](/ml4a/how_neural_networks_are_trained/) had not yet been applied to train neural nets, and thus there was no easy way to optimize neocognitrons or reuse them on different vision tasks.

{% include further_reading.md title="Neocognitron: A Self-organizing Neural Network Model
for a Mechanism of Pattern Recognition
Unaffected by Shift in Position" author="Kunihiko Fukushima" link="http://www.cs.princeton.edu/courses/archive/spr08/cos598B/Readings/Fukushima1980.pdf" %} 

{% include further_reading.md title="Scholarpedia article on neocognitron" author="Kunihiko Fukushima" link="http://www.scholarpedia.org/article/Neocognitron" %} 


## LeNet (1998)

In the late 1980s, [Geoffrey Hinton et al](https://www.nature.com/articles/323533a0) first succeeded in applying backpropagation to the training of neural networks. During the 1990s, a [team at AT&T Labs](https://www.youtube.com/watch?v=FwFduRA_L6Q) led by Hinton's former post-doc student [Yann LeCun](http://yann.lecun.com/) trained a convolutional network, nicknamed ["LeNet"](http://yann.lecun.com/exdb/lenet/), to classify images of handwritten digits to an accuracy of 99.3%. Their system was used for a time to automatically read the numbers in 10-20% of checks printed in the US. LeNet had 7 layers, including two convolutional layers, with the architecture summarized in the below figure.

{% include figure_multi.md path1="/images/figures/lenet.png" caption1="<a href=\"http://yann.lecun.com/exdb/lenet/\">LeNet</a>" %}

Their system was the first convolutional network to be applied on an industrial-scale application. Despite this triumph, many computer scientists believed that neural networks would be incapable of scaling up to recognition tasks involving more classes, higher resolution, or more complex content. For this reason, most applied computer vision tasks would continue to be carried out by other algorithms for more than another decade.

## AlexNet (2012)

Convolutional networks began to take over computer vision -- and by extension, machine learning more generally -- in the early 2010s. In 2009, researchers at the [computer science department at Princeton University](https://www.cs.princeton.edu/), led by [Fei-Fei Li](http://vision.stanford.edu/feifeili/), compiled the [ImageNet database](http://www.image-net.org/), a large-scale dataset containing over [14 million](http://image-net.org/about-stats) labeled images which were manually annotated into 1000 classes using [Mechanical Turk](https://www.mturk.com/mturk/welcome). ImageNet was by far the largest such dataset ever released and quickly became a staple of the research community. A year later, the [ImageNet Large Scale Visual Recognition Challenge (ILSVRC)](http://www.image-net.org/challenges/LSVRC/) was launched as an annual competition for computer vision researchers working with ImageNet. The ILSVRC welcomed researchers to compete on a number of important benchmarks, including classification, localization, detection, and others -- tasks which will be described in more detail later in this chapter. 

{% include figure_multi.md path1="/images/figures/mechanicalturk-imagenet.png" caption1="The <a href=\"https://www.mturk.com/mturk/welcome\">Mechanical Turk</a> backend used to provide labels for ImageNet. Source: <a href=\"https://qz.com/1034972/the-data-that-changed-the-direction-of-ai-research-and-possibly-the-world/\">Dave Gershgorn</a>." %}

For the first two years of the competition, the winning entries all used what were then standard approaches to computer vision, and did not involve the use of convolutional networks. The top-winning entries in the classification tasks had a top-5 error (did not guess the correct class in top-5 predictions) between 25 and 28%. In 2012, a team from the [University of Toronto](http://web.cs.toronto.edu/) led by Geoffrey Hinton, Ilya Sutskever, and Alex Krizhevsky submitted a [deep convolutional neural network nicknamed "AlexNet"](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) which won the competition by a dramatic margin of over 40%. AlexNet broke the previous record for top-5 classification error from 26% down to 15%. 

{% include figure_multi.md path1="/images/figures/alexnet.jpg" caption1="AlexNet (<a href=\"https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)\">original paper</a>)" %}

Since the following year, nearly all entries to ILSVRC have been deep convolutional networks, and classification error has steadily tumbled down to nearly 2% in 2017, the last year of ILSVRC. Convnets [now even outperform humans](https://karpathy.github.io/2014/09/02/what-i-learned-from-competing-against-a-convnet-on-imagenet/) in ImageNet classification! These monumental results have largely fueled the excitement about deep learning that would follow, and many consider them to have revolutionized computer vision as a field outright. Furthermore, many important research breakthroughs that are now common in network architectures -- such as [residual layers](https://arxiv.org/abs/1512.03385) --  were introduced as entries to ILSVRC.

# How convolutional layers work

Despite having their own proper name, convnets are not categorically different from the neural networks we have seen so far. In fact, they inherit all of the functionality of those earlier nets, and improve them mainly by introducing a new type of layer, called a _convolutional layer_, along with a number of other innovations emulating and refining the ideas introduced by neocognitron. Thus any neural network which contains at least one convolutional layer can be called a convolutional network.

## Filters and activation maps

Prior to this chapter, we've just looked at _fully-connected layers_, in which each neuron is connected to every neuron in the previous layer. Convolutional layers break this assumption. They are actually mathematically very similar to fully-connected layers, differing only in the architecture. Let's first recall that in a fully-connected layer, we compute the value of a neuron $z$ as a weighted sum of all the previous layer's neurons, $z=b+\sum{w x}$.

{% include figure_multi.md path1="/images/figures/weights_analogy_2.png" caption1="Weights analogy" %}

We can interpret the set of weights as a _feature detector_ which is trying to detect the presence of a particular feature. We can visualize these feature detectors, as we did previously for MNIST and CIFAR. In a 1-layer fully-connected layer, the "features" are simply the the image classes themselves, and thus the weights appear as templates for the entire classes. 

In convolutional layers, we instead have a collection of smaller feature detectors called _convolutional filters_ which we individually slide along the entire image and perform the same weighted sum operation as before, on each subregion of the image. Essentially, for each of these small filters, we generate a map of responses--called an _activation map_--which indicate the presence of that feature across the image.

The process of convolving the image with a single filter is given by the following interactive demo.

{% include demo_insert.html path="/demos/convolution/" parent_div="post" %}

In the above demo, we are showing a single convolutional layer on an MNIST digit. In this particular network at this layer, we have exactly 8 filters, and below we show each of the corresponding 8 activation maps.

{% include demo_insert.html path="/demos/convolution_all/" parent_div="post" %}

Each of the pixels of these activation maps can be thought of as a single neuron in the next layer of the network. Thus in our example, since we have 8 filters generating $25 * 25$ sized maps, we have $8 * 25 * 25 = 5000$ neurons in the next layer. Each neuron signifies the amount of a feature present at a particular xy-point. It's worth emphasizing the differences in our visualization above to what we have seen before; in prior chapters, we always viewed the neurons (activations) of ordinary neural nets as one long column, whereas now we are viewing them as a set of activation maps. Although we could also unroll these if we wish, it helps to continue to visualize them this way because it gives us some visual understanding of what's going on. We will refine this point in a later section.

Convolutional layers have a few properties, or hyperparameters, which must be set in advance. They include the size of the filters ($5x5$ in the above example), the stride and spatial arrangement, and padding. A full explanation of these is beyond the scope of the chapter, but a good overview of these can be [found here](http://cs231n.github.io/convolutional-networks/).

{% include further_reading.md title="Understanding convolutions" author="Chris Olah" link="http://colah.github.io/posts/2014-07-Understanding-Convolutions/" %} 


## Pooling layers

Before we explain the significance of the convolutional layers, let's also quickly introduce _pooling layers_, another (much simpler) kind of layer, which are very commonly found in convnets, often directly after convolutional layers. These were originally called "subsampling" layers by LeCun et al, but are now generally referred to as pooling.

The pooling operation is used to downsample the activation maps, usually by a factor of 2 in both dimensions. The most common way of doing this is _max pooling_ which merges the pixels in adjacent 2x2 cells by taking the maximum value among them. The figure below shows an example of this.

{% include figure_multi.md path1="/images/figures/max-pooling.png" caption1="Max pooling (source: <a href=\"https://cs231n.github.io/convolutional-networks/\">CS231n</a>)" %}

The advantage of pooling is that it gives us a way to compactify the amount of data without losing too much information, and create some invariance to translational shift in the original image. The operation is also very cheap since there are no weights or parameters to learn.

Recently, pooling layers have begun to gradually fall out of favor. Some architectures have incorporated the downsampling operation into the convolutional layers themselves by using a stride of 2 instead of 1, making the convolutional filters skip over pixels, and result in activation maps half the size. These ["all-convolutional nets"](https://arxiv.org/abs/1412.6806) have some important advantages and are becoming increasingly common, but have not yet totally eliminated pooling in practice.


## Volumes

Let's zoom out from what we just looked at and see the bigger picture. From this point onward, it helps to interpret the data flowing through a convnet as a "volume," i.e. a 3-dimensional data structure. In previous chapters, our visualizations of neural networks always "unrolled" the pixels into a long column of neurons. But to visualize convnets properly, it makes more sense to continue to arrange the neurons in accordance with their actual layout in the image, as we saw in the last demo with the eight activation maps. 

In this sense, we can think of the original image as a volume of data. Let's consider the previous example. Our original image is 28 x 28 pixels and is grayscale (1 channel). Thus it is a volume whose dimensions are 28x28x1. In the first convolutional layer, we convolved it with 8 filters whose dimensions are 5x5x1. This gave us 8 activation maps of size 24x24. Thus the output from the convolutional layer is size 24x24x8. After max-pooling it, it's 12x12x8. 

What happens if the original image is color? In this case, our analogy scales very simply. Our convolutional filters would then also be color, and therefore have 3 channels. The convolution operation would work exactly as it did before, but simply have three times as many multiplications to make; the multiplications continue to line up by x and y as before, but also now by channel. So suppose we were using CIFAR-10 color images, whose size is 32x32x3, and we put it through a convolutional layer consisting of 20 filters of size 7x7x3. Then the output would be a volume of 26x26x20. The size in the x and y dimensions is 26 because there are 26x26 possible positions to slide a 7x7 filter into inside of a 32x32 image, and its depth is 20 because there are 20 filters.

{% include figure_multi.md path1="/images/figures/cnn_volumes.jpg" caption1="Volumes (source: <a href=\"https://cs231n.github.io/convolutional-networks/\">CS231n</a>)" %}

We can think of the stacked activation maps as a sort-of "image."  It's no longer really an image of course because there are 20 channels instead of just 3, as there are in actual RGB images. But it's worth seeing the equivalence of these representations; the input image is a volume of size 32x32x3, and the output from the first convolutional layer is a volume of size 26x26x20. Whereas the values in the 32x32x3 volume simply represent the intensity of red, green, and blue in every pixel of the image, the values in the 26x26x20 volume represent the intensity of 20 different feature detectors over a small region centered at each pixel. They are equivalent in that they are giving us information about the image at each pixel, but the difference is in the quality of the information. The 26x26x20 volume captures "high-level" information abstracted from the original RGB image.

## Things get deep

Ok, here's where things get tricky. In typical neural networks, we frequently have multiple convolutional (and pooling layers) arranged in a sequence. Suppose after our first convolution gives us the 26x26x20 volume, we attach another convolutional layer consisting of 30 new feature detectors, each of which are sized 5x5x20. Why is the depth 20? Because in order to fit, the filters have to have as many channels as the activation maps they are being convolved over. If there is no padding, each of the new filters will produce a new activation map of size 22x22. Since there are 30 of them, that means we'll have a new volume of size 22x22x30. 

How should these new convolutional filters and the resulting activation maps or volume be interpreted? These feature detectors are looking for patterns in the volume from the previous layer. Since the previous layer already gave us patterns, we can interpret these new filters as looking for patterns _of those patterns_. This can be hard to wrap your head around, but it follows straight from the logic of the idea of feature detectors. If the first feature detectors could detect only simple patterns like differently-oriented edges, the second feature detectors could combine those simple patterns into slightly more complex or "high-level" patterns, such as corners or grids. 

And if we attach a third convolutional layer? Then the patterns found from that layer are yet higher-level patterns, perhaps things like lattices or very basic objects. Those basic objects can be combined in another convolutional layer to detect even more complex objects, and so on. This is the core idea behind deep learning: by continuously stacking these feature detectors on top of each other over many layers, we can learn a compositional hierarchy of features, from simple low-level patterns to complex high-level objects. The final layer is a classification, and it is likely to do much better learning from the high-level representation given to it by many layers, than it otherwise would do on the raw pixels or some hand-crafted statistical representation of them alone.

You may be wondering how are the filters determined? Recall that the filters are just collections of weights, just like all of the other weights we've discussed in previous chapters. They are parameters which are learned during the process of training. See the [previous chapter on how neural nets are trained](/ml4a/how_neural_networks_are_trained/) for a review of that process.


## Improving CIFAR-10 accuracy

The following interactive figure shows a confusion matrix of a convolutional network trained to classify the CIFAR-10 dataset, achieving a very respectable 79% accuracy. Although the current state of the art for CIFAR-10 gets around 96%, our 79% result is quite impressive considering that it was trained on nothing more than a [client-side, CPU-only, javascript library](https://cs.stanford.edu/people/karpathy/convnetjs/started.html)! Recall that an ordinary neural network with no convolutional layers [only achieved 37% accuracy](/demos/confusion_cifar/). So we can see the immense improvement that convolutional layers give us. By selecting the first menu, you can see confusion matrices for convnets and ordinary neural nets trained on both CIFAR-10 and MNIST.

{% include demo_insert.html path="/demos/confusion_cifar_convnet/" parent_div="post" %}


## Applications of convnets

Since the early 2010s, convnets have ascended to become the most widely used deep learning algorithms for a variety of applications. Once considered successful only for a handful of specific computer vision tasks, they are now also deployed for audio, text, and other types of data processing. They owe their versatility to the automation of feature extraction, something which was once the most time-consuming and costly process necessary for applying a learning algorithm to a new task. By incorporating feature extraction itself into the training process, it's now possible to re-appropriate an existing convnet's architecture, often with very few changes, into a totally different task or even different domain, and retrain or "fine-tune" it to the new task. 

Although a full review of convnets many applications is beyond the scope of this chapter, this section will highlight a small number of them which are relevant to creative uses.

### In computer vision

Besides for image classification, convnets can be trained to perform a number of tasks which give us more granular information about images. One task closely associated with classification is that of localization: assigning a bounding rectangle for the primary subject of the classification. This task is typically posed as a regression alongside the classification, where the network must accurately predict the coordinates of the box (x, y, width, and height). 

This task can be extended more ambitiously to the task of object detection. In object detection, rather than classifying and localizing a single subject in the image, we allow for the presence of multiple objects to be located and classified within the image. The below image summarizes these three associated tasks.

{% include figure_multi.md path1="/images/figures/localization-detection.png" caption1="Classification, localization, and detection are the building blocks of more sophisticated computer vision systems. Source: <a href=\"http://cs231n.stanford.edu/slides/2016/winter1516_lecture8.pdf\">Stanford CS231n</a>" %}

Closely related is the task of semantic segmentation, a task which involves segmenting an image into all its found objects. This is similar to object detection, but actually demarcates the full border of each found object, rather than just its bounding box.  

{% include figure_multi.md path1="/images/figures/mask-rcnn.jpg" caption1="Mask R-CNN implementing semantic segmentation. Source: <a href=\"https://github.com/facebookresearch/Detectron\">Facebook AI research's Detectron repository</a>" %}

Semantic segmentation and object detection have only become feasible relatively recently. One of the major limitations holding them back previously, besides for the increased complexity compared to single-class classification, was a lack of available data. Even the imagenet dataset, which was used to take classification to the next level, was unable to do anything about detection or segmentation because it had sparse information about the locations of objects. But more recent datasets like [MS-COCO](http://cocodataset.org/) have added richer information for each image into their schema, enabling us to pursue localization, detection, and segmentation more seriously.

Early attempts at training convnets to do multiple object detection typically used a localization-like task to first identify potential bounding boxes, then simply applied classification to all of those boxes, keeping the ones in which it had the most confidence. This approach is very slow  however because it requires at least one forward pass of the network for each of the dozens or even hundreds of candidates. In certain situations, such as with self-driving cars, this latency is obviously unacceptable. In 2016, [Joseph Redmon](https://pjreddie.com/) developed [YOLO](https://pjreddie.com/darknet/yolo/) to address these limitations. YOLO -- which stands for "you only look once" -- restricts the network to only "look" at the image a single time, i.e. it is permitted a single forward pass of the network to obtain all the information it needs, hence the name. It has in some circumstances achieved a 40-90 frames-per-second speed on multiple object detection, making it capable of being deployed in real-time scenarios demanding such responsiveness. The approach is to divide the image into a grid of equally-sized regions, and have each one predict a candidate object along with its classification and bounding box. At the end, those regions with the highest confidence are kept. The figure below summarizes this approach.

{% include figure_multi.md path1="/images/figures/yolo-pipeline.png" caption1="Real-time object detection is possible by training a network to output classifications and localizations for all found objects simultaneously. Source: <a href=\"https://arxiv.org/pdf/1506.02640.pdf\">You Only Look Once: Unified, Real-Time Object Detection (Redmon et al)</a>" %}

{% include figure_multi.md path1="/images/figures/yolo-examples.png" caption1="Some examples of YOLO detecting objects in image. Source: <a href=\"https://arxiv.org/pdf/1612.08242.pdf\">YOLO9000: Better, Faster, Stronger (Redmon)</a>" %}

Still more tasks relevant to computer vision have been introduced or improved in the last few years, as well as systems specialized for retrieving text from images by combining convnets with recurrent neural networks (to be introduced in a later chapter). Another class of tasks involves annotating images with natural language by combining convnets with recurrent neural networks. This chapter will leave those to be discussed by future chapters, or within the included links for further reading. 

### Audio applications

Perhaps one of the most surprising aspects about convnets is their versatility, and their success in the audio domain. Although most introductions to convnets (like this chapter) emphasize computer vision, convnets have been achieving state-of-the-art results in the audio domain for just as long. They are routinely used for [speech recognition](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.224.2749&rep=rep1&type=pdf) and other audio information retrieval work, supplanting older approaches over the last few years as well. Prior to the ascendancy of neural networks into the audio domain, speech-to-text was typically done using a [hidden markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model) along with handcrafted audio feature extraction done using conventional [digital signal processing](https://en.wikipedia.org/wiki/Digital_signal_processing).

A more recent use case for convnets in audio is that of [WaveNet](https://deepmind.com/blog/wavenet-generative-model-raw-audio/), introduced by researchers at [DeepMind](https://deepmind.com/) in late-2016. WaveNets are capable of learning how to synthesize audio by training on large amounts of raw audio. WaveNets have been used by [Magenta](https://magenta.tensorflow.org) to create custom [digital audio synthesizers](https://magenta.tensorflow.org/nsynth) and are now used to generate the voice of [Google Assistant](https://deepmind.com/blog/wavenet-launches-google-assistant/). 

{% include figure_multi.md path1="/images/figures/CD-CNN-HMM.png" caption1="Diagram depicting CD-CNN-HMM, an architecture used for speech recognition. The convnet is used to learn features from a waveform's spectral representation. Source: <a href=\"http://recognize-speech.com/acoustic-model/knn/comparing-different-architectures/convolutional-neural-networks-cnns\">Speech Recognition Wiki</a>" path2="/images/figures/wavenet.gif" caption2="WaveNets are used to create a generative model for probabilistically producing audio one sample at a time. Source: <a href=\"https://deepmind.com/blog/wavenet-generative-model-raw-audio/\">DeepMind</a>" %}

Generative applications of convnets, including those in the image domain and associated with computer vision, as well as those that also make use of recurrent neural networks, are left to future chapters. 

{% include further_reading.md title="Object Localization and Detection" author="Leonardo Araujo dos Santos" link="https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/object_localization_and_detection.html" %} 

{% include further_reading.md title="Conv Nets: A Modular Perspective" author="Chris Olah" link="https://colah.github.io/posts/2014-07-Conv-Nets-Modular/" %} 

{% include further_reading.md title="Visualizing what ConvNets learn (Stanford CS231n" author="Andrej Karpathy" link="https://cs231n.github.io/understanding-cnn/" %} 

{% include further_reading.md title="How do Convolutional Neural Networks work?" author="Brandon Rohrer" link="https://brohrer.github.io/how_convolutional_neural_networks_work.html" %} 

{% include further_reading.md title="Convnet visualization demo" author="Adam Harley" link="http://scs.ryerson.ca/~aharley/vis/conv/" %} 
