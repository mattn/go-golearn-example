package main

import (
	"fmt"
	"log"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
)

func main() {
	data, err := base.ParseCSVToInstances("datasets/iris.csv", false)
	if err != nil {
		log.Fatal(err)
	}

	classifier := knn.NewKnnClassifier("euclidean", "linear", 2)

	trainData, testData := base.InstancesTrainTestSplit(data, 0.50)
	classifier.Fit(trainData)

	pred, err := classifier.Predict(testData)
	if err != nil {
		log.Fatal(err)
	}
	confusionMat, err := evaluation.GetConfusionMatrix(testData, pred)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(evaluation.GetSummary(confusionMat))
}
