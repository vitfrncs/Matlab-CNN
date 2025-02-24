%Armazenando caminhos das pastas em veriáveis 
outputFolder = fullfile('imagens');
rootFolder = fullfile(outputFolder, '101_ObjectCategories');

%Teremos três categorias de dados de entrada
categories = {'electric_guitar', 'brain', 'laptop'};
 
%image data store: 
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource','foldernames');
%possui dois argumentos: localização das imagens e (nome e valor)

tbl = countEachLabel(imds);
%segundo conuntEachLabel, não há o mesmo n° de imagens para cada categoria
%precisamos que esse número seja o mesmo.
 
%descobrindo a categoria com menos imagens:
%focando na segunda coluna de tbl
catMinima = min(tbl{:,2});

%rediziremos o número o número de imagens de cada catgoria para o mínimo:
imds = splitEachLabel(imds, catMinima, 'randomize');

electric_guitar = find(imds.Labels == 'electric_guitar', 1);
brain = find(imds.Labels == 'brain', 1);
laptop = find(imds.Labels == 'laptop',1);

%figure;
%subplot(2,2,1);
%imshow(readimage(imds,electric_guitar));
%subplot(2,2,2);
%imshow(readimage(imds,brain));
%subplot(2,2,3);
%imshow(readimage(imds,laptop));

%as imagens foram carregadas
%agora, chamaremos uma CNN pré treinada. usaremos ResNet50

net = resnet50();
%Mostra a arquitetura da ResNet50 :D
%figure
%plot(net)
%title('Arquitetura de ResNet-50')
%set(gca, 'YLim', [150 170]);

%separando as imagens para treinar a rede neural
[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomized');

%trnsformando o tamanhos e cores das imagens
tamImagem = net.Layers(1).InputSize;
augmetedTrainingSet = augmentedImageDatastore(tamImagem, ...
    trainingSet, 'ColorPreprocessing','gray2rgb');
augmetedTestSet = augmentedImageDatastore(tamImagem, ...
    testSet, 'ColorPreprocessing','gray2rgb');

%peso1 é uma matriz e diz o peso da segunda camada de convolução
peso1 = net.Layers(2).Weights;
%transformando a matriz de peso em uma imagem:
peso1 = mat2gray(peso1);

%figure
%montage(peso1)
%title("Peso da primeira camada convolucional")

featureLayer = 'fc1000';
trainingFeatures = activations(net, ...
    augmetedTrainingSet, featureLayer, 'MiniBatchSize',32, 'OutputAs','columns');


trainingLables = trainingSet.Labels;
%função que retorna um modelo treinado:
classificador = fitcecoc(trainingFeatures, trainingLables, 'Learner', ...
    'Linear', 'onevsall', 'Observation', 'columns');

%extraindo as caracteristicas das imagens de teste:
testFeatures = activations(net, ...
    augmetedTestSet, featureLayer, 'MiniBatchSize',32, 'OutputAs','columns');

%medindo a acuracia do classificador treinado:
%a função predict retorna um vetor de classes previstas baseadas em classes
%treinadas do classificador
predictLabels = predict(classificador, testFeatures, 'ObservationsIn', 'columns');

%a variavel testLables guarda as verdadeiras classificações das imagens
testLables = testSet.Labels;

%returns the confusion matrix C determined by the known and predicted
%groups in group and grouphat, respectively: (documentação)
confMat = confusionmat(testLables, predictLabels);
confMat = bsxfun(@rdivide, confMat, sum(confMat,2));

mean(diag(confMat));

%Testando nova imagem:
newImage = imread(fullfile('guitarra.png'));

ds = augmentedImageDatastore(tamImagem, newImage, 'ColorPreprocessing', 'gray2rgb');

imageFeatures = activations(net, ...
    ds, featureLayer, 'MiniBatchSize',32, 'OutputAs','columns');

label = predict(classificador, imageFeatures, 'ObservationsIn', 'columns');

sprintf('A imagem pertence a classe %s', label);
