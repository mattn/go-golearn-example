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

	net := neural.NewMultiLayerNet([]int{3})
	net.MaxIterations = 20000
	net.Fit(data)
	pred := net.Predict(data)
	fmt.Println(pred)
}
