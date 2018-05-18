// Learn more about F# at http://fsharp.org
//XPlot.Plotly.WPF

open System
open PLplot
open MathNet.Numerics.Distributions
open MathNet.Numerics.Random
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open System.Net.NetworkInformation

type ErrType = string

type ActFn = | ID    //identity f(x) = x
             | TANH // f(x) = tanh(x)
              //| RELU

type Layer = {wMatx: Matrix<float>; actFn: ActFn;}

type Network = Layer list

type NodesVect = {sVect: Vector<float>; xVect: Vector<float>} //each nodesVect will be S => without actFn and X => with actFn

type NodesVectList = NodesVect list

let epochs = 2000//50
let lr = 0.03//0.0001

let getActFn (fn: ActFn) : (float->float) =
    match fn with
    | ID -> id
    | TANH -> tanh

let getActFnDeriv (fn: ActFn) : (float->float) =
    match fn with
    | ID -> (fun _ -> 1.0)
    | TANH -> (fun x  -> 1.0 - (tanh(x)**2.0))

//Only works with MSE as metric, shuld be called last layer deltaL! or error delta! or err deriv!
let lastLayerDeriv (x_L : Vector<float>) (s_L : Vector<float>) (y : Vector<float>) (theta_L' : ActFn) =
    (2.0*(x_L - y)).PointwiseMultiply(s_L.Map (Func<float, float> (getActFnDeriv (theta_L'))))

//this is the SQUARE ERROR PART OF MSE, THE MEAN PART IS FOUND WHEN DOING THIS FOR ALL X_train or ALL X_test
//notice this gives feature-wise square-error! there is no "mean" here.
// to get mean we do mean feature-wise only! so we get a mse Vector afterwards where each component is mean over all data points
//we will print out error separately for each output feature.
//we CAN also either find the avg of avg error over all features to get one single scalar val i.e. mean of MSE
//what we will actually do and this is equivalent as above line is
//do sq err vector, find its mean so this is mean of errors for all features.
//then mean this over all data points.
let meanFeatureSqErr (h : Vector<float>) (y: Vector<float>) =
            // (h - y).PointwisePower(2.0).SumMagnitudes() / (float) (h - y).Count
            //L2Norm gives: (h_1 - y_1)^2 + (h_2 - y_2)^2 + ... + (h_d - y_d)^2
            // just divide by length to get mean
            // len of h and y must be same!!
            // a mean of the feature error vect squared!
            (h - y).PointwisePower(2.0).SumMagnitudes() / (float) h.Count
           


//----helper func for dataset---->
let yRealFn x = 0.4*(x**2.0)+0.2+0.3*x*sin(8.0*x)
let yRealFnAndNoise x = (yRealFn x) + 0.02*Normal.Sample(0.0, 1.0)
//provide start, step, end and yFn
//return a tuple array with each element as (x_i, y_i) //we return an array so we can do in-place sort for random shiffling
//we can then convert it into a list for list reduction although array reduction is also possible.
let genDataSet (startVal) (endVal) (stepVal) (fN : 'T -> 'T) = 
    [| startVal .. stepVal .. endVal|]
    |> Array.map(fun x -> (CreateVector.Dense(1, x), CreateVector.Dense(1, fN x)))
    


let initNetwork (nodeLst : int list) (actFnLst : ActFn list) : Result<Network, ErrType> =
    //e.g ([14 ; 32 ; 8 ; 1], [relu, relu, identity]) so list2 must be 1 less or return result.error
    //return [ (wmtx1, actFnLayer1) ... (wmatxL, actFnLayerL) ]
    let genRandMatx (rows) (cols) : Matrix<float> = 
        CreateMatrix.Random(rows,cols)

    match (nodeLst.Length - actFnLst.Length) with
    | 1 -> 
        match nodeLst.Length  with
        | x when x > 1 ->
            match (List.length (List.filter (fun x -> x > 0) nodeLst)) with
            | y when y = x ->
                List.init (actFnLst.Length) (fun i ->
                    {
                        wMatx = genRandMatx (nodeLst.[i] + 1) (nodeLst.[i+1]);
                        actFn = actFnLst.[i];
                    }
                )
                |> Ok
            | _ -> Error("The specification was incorrect, all item in nodeLst must be positive.")
        | _ -> Error("The specification was incorrect, need at-least two items in nodeLst.")
        //For each w_ij in WMatx, 0 < i < d^{l-1} ; 1 < j < d^{l}
        //biases for each node in layer l will be added as extra node in layer l-1 hence i>=0
        //i rows ; j cols
    | _ -> Error("The specification was incorrect, nodeLst must be exactly one item greater than actFnLst.")


let fwdProp (x: Vector<float>) (net : Network) : (NodesVectList * Network) =
    // take in a vector of type float this is our input can be mx1 for any m dim of input
    // also takes in a network that will get reduced
    // finally returns a vector which will be our hypothesis h(x) can be of any dim matching y dim
    //actFn((w.T * x + b))

    let propNetwork (nvectLstAndLayers: NodesVectList * Network) (layer : Layer) : (NodesVectList * Network) =
        
        let nvectLst, layerLst = nvectLstAndLayers
         //result of matmul will be nx1 matx where n=l^(d) so excluding the "1" node or x_0^l(d)
        let newSVect =  layer.wMatx.Transpose() * CreateVector.Dense((Array.concat [ [|1.0|] ; nvectLst.Head.xVect.ToArray() ]))  //bias is auto included by weight[0, :]
        let newXVect = (newSVect.Map (Func<float, float> (getActFn layer.actFn)))   //Func type-casting needed by vector
        ({xVect=newXVect ; sVect=newSVect} :: nvectLst , layer :: layerLst)

    (List.fold propNetwork ([{xVect=x ; sVect=x}], []) net)   //initially sVect == xVect i.e. no actFn for raw input


let backProp (yVect : Vector<float>) (fwdPropEval : (NodesVectList * (Network))) : Network =

    let rec _calcPrevDelta (deltaLst : Vector<float> list) (layerLst: Network) (nodesInLayerLst: NodesVect list) : Vector<float> list =
        match layerLst with
        | [_] -> deltaLst
        | head :: head2 :: _ -> 
                    //printfn "****DEBUG PREVDELTA: %A\n ***DEBUG wMatx: %A, \nDEBUG: nodesVect: %A" deltaLst.Head head.wMatx nodesInLayerLst
                    let interim = head.wMatx.RemoveRow(0) * deltaLst.Head

                    let theta' = nodesInLayerLst.Head.sVect.Map (Func<float, float> (getActFnDeriv (head2.actFn)))
                    //printfn "***INTERIM --> \n\n%A\n\nTHETA' --> \n\n%A\n\n" interim theta'

                    //hamadard product
                    let prevDelta = interim.PointwiseMultiply(theta')
                    _calcPrevDelta (prevDelta :: deltaLst) (layerLst.Tail) (nodesInLayerLst.Tail)
        | [] -> [] //should throw error but this wil never hit if wMatx is checked initially during nnArch
    
    let layerListUpdater (prevNodes : NodesVect) (currDelta : Vector<float>) (currLayer : Layer) : Layer =
        //must add 1.0 on top for correct shape. toRowMatx is same as (colVect).T
        let dEdWMatx =  CreateMatrix.DenseOfColumnArrays( (Array.concat [ [|1.0|] ; prevNodes.xVect.ToArray() ])  ) * currDelta.ToRowMatrix()
        let newWMatx = currLayer.wMatx - (lr*dEdWMatx)

        // printfn "***updatedW: \n%A\n" newWMatx
        // printfn "dEdWMATx: \n%A\n" dEdWMatx
        {currLayer with wMatx=newWMatx}
        

    let xAndSLstRev, netRev = fwdPropEval

    // first step is to calculate last layer error
    // this uses a combination of derivative of mean square error and derivative of last layer act Fn
    //so u foldback and calc dleta and every time you get new delta u append to head so that is O(1)
    //A list of all error deltas in reverse order
    //first arg must be a list
    //slice the xAndSList cause we need the prev elements for multiply
    let deltaL = lastLayerDeriv xAndSLstRev.Head.xVect xAndSLstRev.Head.sVect yVect netRev.Head.actFn
    let deltaRevLst = _calcPrevDelta [deltaL] netRev xAndSLstRev.Tail |> List.rev//List.fold2 calcPrevDelta [deltaL] netRev subRevLst 

    //now simply use list fold2 to calc Wmatx from deltaRevLst and xAndSRevLst.Tail
    //also since they r rev if for every wMatx you append to front, then in the end your new wMAtxList will be automatically sorted
    
    //printfn "***DEBUG_FINAL_LAYER_ERR_DERIV: %A" deltaL
    //printfn "***THE DELTAS: \n\n%A" deltaRevLst
    List.map3 layerListUpdater xAndSLstRev.Tail deltaRevLst netRev |> List.rev

// evaluate avg network error by finding the mean of fwdProp errors on a specific data-set
// eval array the data points to test on
// evalnet: the trained network
//the errFn to supply will return a single float value for multiple output features and for now this will be the mean of the sq err of features
let evalAvgNetworkErr (xAndyEvalArr : (Vector<float> * Vector<float>) []) (errFn) (evalNet) =
        let perfGradDesc (xAndyTuple : (Vector<float> * Vector<float>)) : float =
            let x, y = xAndyTuple
            evalNet 
            |> fwdProp (x) //?? calc errr using err fn and accumulate
            |> function
               | nVLst, _ -> errFn nVLst.Head.xVect y
        
        //return the avg error over ALL data points
        (Array.sumBy perfGradDesc xAndyEvalArr) / (float) xAndyEvalArr.Length


let train (xAndyTrainArr) (xAndyTestArr) (initNetwork : Network) (epochs) : (float list) * Network =
    let trainAndEvalEpoch  (xAndyTrainArrAndNetwork : (((Vector<float> * Vector<float>) []) * Network)) (epoch : int) : (float *  ((Vector<float> * Vector<float>) [] * Network)) =   
        // for every epoch u will have an initial state which will be xAndyArr * initNetwork
        // u will create permutation of this array and then u will feed this to tranEpoch fn.
        // once u get result u return shuffledXAndY and new weights but u also have a map part
        // the map part will calc MSE which is l2normerror or sqerr for all X_train
        // note here we are not concerned about generalisation but the ability of the network to be a universal funcion approximator!
        // however this same mse fn can be used for test data later on
        let trainEpoch (_xAndyTrainArrAndInitNet) : (Vector<float> * Vector<float>) [] * Network =
            let perfGradDesc (currNetwork : Network) (xAndyTuple) : Network =
                currNetwork |> fwdProp (fst xAndyTuple) |> backProp (snd xAndyTuple)
            
           
            _xAndyTrainArrAndInitNet
            |> function
               | (_xAndyTrainArr, _initNet) 
                 -> Combinatorics.SelectPermutationInplace(_xAndyTrainArr) ;
                    _xAndyTrainArr, (Array.fold perfGradDesc (_initNet) (_xAndyTrainArr))
            
            //return new network with shuffled array
           
        
        //printfn "Training for epoch: %A" epoch
        xAndyTrainArrAndNetwork
        |> trainEpoch
        |> function 
           | xAndyShuffArr, newNet -> evalAvgNetworkErr xAndyTestArr meanFeatureSqErr newNet
                                      |> (fun err -> (err, (xAndyShuffArr, newNet))) 

    
    [1 .. epochs]
    |> List.mapFold trainAndEvalEpoch (xAndyTrainArr, initNetwork)
    |> function
       | err, (_ , finalNet) -> err, finalNet







let plot xAndyTrueArr xAndyTrainArr =
    use pl = new PLStream()
    PLplot.Native.sdev("wincairo")
    pl.init()
    pl.env( 0.0, 1.0, 0.0, 1.0, AxesScale.Independent, AxisBox.BoxTicksLabelsAxes )
    let xTrueArr, yTrueArr = Array.unzip xAndyTrueArr
    let xTrainArr, yTrainArr = Array.unzip xAndyTrainArr
    pl.lab( "x ->", "y ->", (sprintf "A plot of true values ('x') vs final prediction (line) [MSE: %0.4f]" 1.0))
    pl.line(xTrueArr, yTrueArr)
    pl.poin( xTrainArr, yTrainArr, 'o')

[<EntryPoint>]
let main argv = 
    //optimise weights, print final results and then plot the results

    //printfn "ID(-343434.0) == %A" ((getActFn ID) -343434.0)

    // printfn "%A" "test"
    // |> printfn "%A"

    //***test code
    //(initNetwork [ 1 ; 1 ] [ ID ]) |> printfn "Result:\n%A"

    (*FOR DEBUGGING USING FIXED VALUES*)
    // // let x = 0.2
    // // let y = yRealFn x //temp for now we only do without noise for debugging
    // // printfn "Y_VAL****: %A" y
    // // let w_l1 = DenseMatrix.OfRowArrays([| [|0.1 ; 0.15|] ; [|0.05 ; 0.2|] |])
    // // let w_l2 = DenseMatrix.OfRowArrays([| [|0.7|] ; [|-0.6|] ; [|0.4|] |])
    // // let debugNNArch = [ {wMatx=w_l1 ; actFn=TANH} ; {wMatx=w_l2 ; actFn=ID} ]

    // // printfn "***DEBUG_ARCH: %A\n" debugNNArch

    // // debugNNArch
    // // |> fwdProp (CreateVector.Dense([|x|]))
    // // |> backProp (CreateVector.Dense([|y|]))
    // // |> printfn "***DEBUG_BACK-PROP_RESULT: \n\n%A" //|> //printfn "***DEBUG_FWD-PROP_RESULT: \n\n%A"

    //***Real code
    // let fwdPropResult = 
    //     nnArch
    //     |> fwdProp (CreateVector.Dense([|1.0|]))
    // match fwdPropResult with
    // | Ok(x) -> now do all folds (see below)!! -> plot -> print final weight
    // | Error(x) -> now print error


    let xAndyTrueArr = genDataSet 0.0 1.0 0.001 yRealFn
    let xAndyTrainArr = genDataSet 0.0 1.0 0.01 yRealFnAndNoise


    // to be done for every epoch!!
    //Combinatorics.SelectPermutationInplace (xAndyTrainArr)
    //let trr = (train xAndyTrainArr )

    (initNetwork [ 1 ; 8 ; 1] [TANH ; ID])
    |> function
       | Ok(x) -> (train xAndyTrainArr xAndyTrainArr x epochs) |> (fun (errLst, finalNet) -> 
                     (printfn "Training for %A epochs done!\nFinal Result:\n%A\n\nFinal Errors:\n%A" epochs finalNet (List.rev errLst)))
       | Error(x) -> printfn "%A" x

    //printfn "The shufleddd array: \n%A" xAndyTrainArr

    let l = 
        let mapper (a : (Vector<float> * Vector<float>)) : float = 
            a
            |> function
               | x, y -> x.Item(0)
        Array.map mapper xAndyTrueArr
    
    //plot xAndyTrueFloats xAndyTrueFloats




    (*
        top level:
        make a list of epochs
        epochLst = [0 .. epochs]

        List.fold epoch with initial weigghts

        in epoch folder function

        get SelectPermutation from array of x,y pairs that u initially created for training at top level
        Array.fold each x,y pair

        now nested folds!
        for inner fold do fwdprop and backprop get w then pass this as new state
        at end pass newW to carry onwards to outer fold then pass on for every epoch.
        in end get the same training list now map it using const weights to h values
        now plot h values!! with orig y vals!
    *)

    0 // return an integer exit code
