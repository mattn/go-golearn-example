package main

import (
	"fmt"
	"log"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/neural"
)

func main() {
	data, err := base.ParseCSVToInstances("datasets/xor.csv", false)
	if err != nil {
		log.Fatal(err)
	}

	nn := neural.NewMultiLayerNet([]int{3})
	nn.MaxIterations = 20000
	nn.Fit(data)
	pred := nn.Predict(data)
	fmt.Println(pred)
}
