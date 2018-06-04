open System
open PLplot
open MathNet.Numerics.Distributions
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics

type ErrType = string

type ActFn = | ID    //identity f(x) = x
             | TANH // f(x) = tanh(x)
              //| RELU

type Layer = {wMatx: Matrix<float>; actFn: ActFn;}

type Network = Layer list

type NodesVect = {sVect: Vector<float>; xVect: Vector<float>} //each nodesVect will be S => without actFn and X => with actFn

type NodesVectList = NodesVect list


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
//what we will actually do and this is equivalent as above line is
//do sq err vector, find its mean so this is mean of errors for all features.
//then mean this over all data points.
let meanFeatureSqErr (h : Vector<float>) (y : Vector<float>) =
            (h - y).PointwisePower(2.0).SumMagnitudes() / (float) h.Count
           


//----helper func for dataset---->
let yRealFn x = -0.1*(log10(x+0.1))+0.4+0.5*x*cos(12.0*x)
let yRealFnAndNoise x = (yRealFn x) + 0.02*Normal.Sample(0.0, 1.0)
//provide start, step, end and yFn
//return a tuple array with each element as (x_i, y_i) //we return an array so we can do in-place sort for random shiffling
//we can then convert it into a list for list reduction although array reduction is also possible.
let genDataSet1D (startVal) (endVal) (stepVal) (fN : float -> float) = 
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


let backProp (learnRate: float) (yVect : Vector<float>) (fwdPropEval : (NodesVectList * (Network))) : Network =

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
        let newWMatx = currLayer.wMatx - (learnRate*dEdWMatx)

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
let evalAvgEpochErr (xAndyEvalArr) (errFn) (evalNet) =
        let hAndErrEval (xAndyTuple) =
            let x, y = xAndyTuple
            evalNet 
            |> fwdProp (x) //?? calc errr using err fn and accumulate
            |> function
               | nVLst, _ -> ((x, nVLst.Head.xVect), errFn nVLst.Head.xVect y)
        
        //return ([h(x1)..h(xN)], avgErr) for all x1..xN samples
        //the error per sample is an average over each output feature so its a float
        //now avg this error over ALL data points
        //return an array of h for all data-set and also the avg err over all of them
        Array.map hAndErrEval xAndyEvalArr
        |> Array.unzip
        |> function
           | xAndhArr, errArr -> (xAndhArr, (Array.sum errArr / (float) xAndyEvalArr.Length))


let train (xAndyTrainArr) (initNetwork) (epochs) (learnRate) : (float list) * Network =
    let trainAndEvalEpoch  (xAndyTrainArrAndNetwork) (epoch)  =   
        // for every epoch u will have an initial state which will be xAndyArr * initNetwork
        // u will create permutation of this array and then u will feed this to tranEpoch fn.
        // once u get result u return shuffledXAndY and new weights but u also have a map part
        // the map part will calc MSE which is l2normerror or sqerr for all X_train
        // note here we are not concerned about generalisation but the ability of the network to be a universal funcion approximator!
        let trainEpoch (_xAndyTrainArrAndInitNet) =
            let perfGradDesc (currNetwork : Network) (xAndyTuple) : Network =
                currNetwork |> fwdProp (fst xAndyTuple) |> backProp (learnRate) (snd xAndyTuple)
            
           
            _xAndyTrainArrAndInitNet
            |> function
               | (_xAndyTrainArr, _initNet) 
                 -> Combinatorics.SelectPermutationInplace(_xAndyTrainArr) ;
                    _xAndyTrainArr, (Array.fold perfGradDesc (_initNet) (_xAndyTrainArr))
            
            //return new network with shuffled array
           
        //we use xAndyTrainArr to evaluate our error in the end of every epoch
        //in future we can also use xAndyTestArr and return tuples of errors
        //printfn "Training for epoch: %A" epoch
        xAndyTrainArrAndNetwork
        |> trainEpoch
        |> function 
           | xAndyShuffArr, newNet -> evalAvgEpochErr xAndyTrainArr meanFeatureSqErr newNet
                                      |> (fun (_, err) -> (err, (xAndyShuffArr, newNet))) 

    //main entry-point
    [1 .. epochs]
    |> List.mapFold trainAndEvalEpoch (xAndyTrainArr, initNetwork)
    |> function
       | errLst, (_ , finalNet) -> errLst, finalNet


//extract the 0th dim from an array of N-dim vector tuples (x, y)
// only used when you need to plot data
let extractFstDim xAndyVectArr =
    let mapper (a : (Vector<'a> * Vector<'a>))= 
            a  |> (fun (x, y) -> x.[0], y.[0])
    Array.map mapper xAndyVectArr




let plot xAndyTrueArr xAndyDataArr xAndhTrainArr xAndhTestArr (errArr : float []) testErr xMax yMax =
    use pl = new PLStream()
    PLplot.Native.sdev("png")
    pl.init()
    pl.ssub(1, 2)
    pl.col0(1)
    pl.env( 0.0, xMax, 0.0, yMax, AxesScale.Independent, AxisBox.BoxTicksLabelsAxesMajorGrid )
    let xTrueArr, yTrueArr = Array.unzip xAndyTrueArr
    let xDataArr, yDataArr = Array.unzip xAndyDataArr
    let xTestArr, hTestArr = Array.unzip xAndhTestArr   //the x values shuld be the same as xDataArr so that its like for like comparision
    let xTrainArr, hTrainArr = Array.unzip xAndhTrainArr
    
    pl.col0(2)
    pl.lab( "x ->", "y ->", (sprintf "True Fn (line), Dataset ('.'), Train-Eval ('o'), Test-Eval ('x') [Train-MSE: %0.4f, Test-MSE: %0.4f, Epochs: %d]" errArr.[errArr.Length - 1] testErr errArr.Length))
    pl.col0(5)
    pl.line(xTrueArr, yTrueArr)             // underlying function (excludes noise from data-set)
    pl.col0(11)
    pl.poin( xDataArr, yDataArr, '.')       // full data-set for training and testing
    pl.col0(5)
    pl.poin( xTestArr, hTestArr, 'x')       // evaluation on unseen data (test)
    pl.col0(3)
    pl.poin( xTrainArr, hTrainArr, 'o')     // evaluation on seen data (train)

    pl.col0(1)
    pl.env( 1.0, float(errArr.Length), 0.0, Array.max errArr, AxesScale.Independent, AxisBox.BoxTicksLabelsAxesMajorGrid )
    pl.col0(2)
    pl.lab( "Epoch ->", "MSE ->", "Training Loss")
    pl.col0(3)
    pl.line( [| 1.0 .. float(errArr.Length) |], errArr)



[<EntryPoint>]
let main argv = 
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
    
    printfn "Training vanilla NN with stochastic gradient descent..."
    let stopWatch = System.Diagnostics.Stopwatch.StartNew()
    let xAndyTrueArr = genDataSet1D 0.0 1.0 0.001 yRealFn
    let xAndyDataArr = genDataSet1D 0.0 1.0 0.01 yRealFnAndNoise
    
    let xAndyTestArr  = Array.filter ( fun (x : Vector<float>, _) -> x.[0]*100.0 % 20.0 = 0.0) xAndyDataArr
    let xAndyTrainArr  = Array.filter ( fun (x : Vector<float>, _) -> x.[0]*100.0 % 20.0 <> 0.0) xAndyDataArr
    
    let epochs = 3000   //2500//50 //2500 for <3s at times
    let lr = 0.03       //0.0001

    printfn "Dataset Length: %A\tTrain-set Length: %A\tTest-set Length: %A\tLearn-rate: %A\tEpochs: %d" xAndyDataArr.Length xAndyTrainArr.Length xAndyTestArr.Length lr epochs

    (initNetwork [ 1 ; 8 ; 1] [TANH ; ID])
    |> function
        | Ok(net) -> (train xAndyTrainArr net epochs lr)
                     |> function
                        | (errLst, finalNet) -> 
                          printfn "Training for %A epochs done! Took: %dms" epochs stopWatch.ElapsedMilliseconds
                          //printfn "Final Network:\n%A\n\nFinal Train Error Lst Reversed:\n%A" finalNet (List.rev errLst)
                          
                          let xAndhTestArr, avgTestErr = evalAvgEpochErr (xAndyTestArr) (meanFeatureSqErr) (finalNet)
                          let xAndhTrainArr, _ = evalAvgEpochErr (xAndyTrainArr) (meanFeatureSqErr) (finalNet)
                          
                          plot (extractFstDim xAndyTrueArr) (extractFstDim xAndyDataArr) (extractFstDim xAndhTrainArr) (extractFstDim xAndhTestArr) (Array.ofList errLst) avgTestErr 1.0 1.0
                     
        | Error(x) -> printfn "%A" x


    0 // return an integer exit code
