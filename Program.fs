// Learn more about F# at http://fsharp.org
//XPlot.Plotly.WPF

open System
open PLplot
open MathNet.Numerics.Distributions
open MathNet.Numerics.Random
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Storage
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra
open System.Net
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra

type ErrType = string

type Weight = {w1: float; w2: float; b: float}

// type ActFn = ActFn of (float->float)
// type ActFnDeriv = ActFnDeriv of (float->float)
(*
    try using pattern matching for multiple fn under actfn then u only need to specify
    actfn list then seperate fn can pattern match for ApplyActFn andApplyActFnDeriv
*)

    type ActFn =    | ID    //identity f(x) = x
                    | TANH // f(x) = tanh(x)
                    //| RELU

// type ActFnDeriv =   | ID' // f(x) = 1
//                     | TANH' // f(x) = 1 - (tanh(x))^2
                    //| RELU' of (float->float)
//type ActFnDeriv = (float->float)

type Layer = {wMatx: Matrix<float>; actFn: ActFn;}

type Network = {lLst: Layer list;}

type NodesVect = {sVect: Vector<float>; xVect: Vector<float>} //each nodeVect will be S => without actFn and X => with actFn

type NodesVectList = NodesVect list

//make a tuple of nodes,actFn,
//list.map this and u get weight matrix+actFn tuple list.
//make this list only for hidden layers
//for input and output layer have inputFeatureCount, outputFeatureCount??
//so [ 14, [] , (1, identityFn)]
//or easier method:
// [(14,_) ; (32, relu) ; (8, relu) ; (1, ident)]

//or ([14 ; 32 ; 8 ; 1], [relu, relu, identity]) so list2 must be 1 less or else return result.error

//return is [ (wmtx1, actFnLayer1) ... (wmatxL, actFnLayerL) ]

(*
    see auto diff lib f# for ideas 
    fwd prop:
    assume single input for now from sample => x_0
    input x_0=0.5 then reduce weight matrix list by actFn((w.T * x + b)) => x_1
    do reduce to get final output
    compare this with fixed weight example test

    back_prop
    do delta_x:

    again either do back reduce or map or simply recursion
    do cleaver way to update weights maybe use map or something

    create main list and reduce it by applying fwd+back to every item and returning new weight set.
    in end u have pass through all but u must randomise again
*)

// let m : Matrix<float> = CreateMatrix.Random(8,3)
// printfn "The matrix: \n%A" m

let epochs = 50
let lr = 0.0001

let yRealFn x = 0.4*(x**2.0)+0.2+0.3*x*sin(8.*x)
let yRealFnAndNoise x = (yRealFn x) + 0.2*Normal.Sample(0.0, 1.0)

let getActFn (fn: ActFn) : (float->float) =
    match fn with
    | ID -> id
    | TANH -> tanh

let getActFnDeriv (fn: ActFn) : (float->float) =
    match fn with
    | ID -> (fun _ -> 1.0)
    | TANH -> (fun x  -> 1.0 - (tanh(x)**2.0))
    //| RELU(_) -> 

//Only works with MSE as metrics!
let lastLayerDeriv (x_L : Vector<float>) (s_L : Vector<float>) (y : Vector<float>) (theta_L' : ActFn) =
    (2.0*(x_L - y)).PointwiseMultiply(s_L.Map (Func<float, float> (getActFnDeriv (theta_L'))))
    
    //2.0*(x_L - y)*( s_L.Map (Func<float, float> (getActFnDeriv (theta_L')))  )



//provide start, step, end and yFn
//return a tuple list with each element as (x_i, y_i)
let genDataSet (startVal) (endVal) (stepVal) (fN: 'a -> 'b) = 
    [ startVal .. stepVal .. endVal]
    |> List.map(fun x -> (x, yRealFnAndNoise x))
    

let genRandMatx (rows) (cols) : Matrix<float> = 
    CreateMatrix.Random(rows,cols)


let initNetwork (nodeLst : int list) (actFnLst : ActFn list) : Result<Network, ErrType> =
    //or ([14 ; 32 ; 8 ; 1], [relu, relu, identity]) so list2 must be 1 less or else return result.error
    //return is [ (wmtx1, actFnLayer1) ... (wmatxL, actFnLayerL) ]

    //_initNetwork(ndLst, actLst, netLst)

    match (nodeLst.Length - actFnLst.Length) with
    | 1 -> 
        match nodeLst.Length with
        | x when x > 1 ->
            List.init (actFnLst.Length) (fun i ->
                {
                    wMatx = genRandMatx (nodeLst.[i] + 1) (nodeLst.[i+1]);
                    actFn = actFnLst.[i];
                }
            )
            |> (fun layers -> { lLst = layers; } )
            |> Ok
        | _ -> Error("The specification was incorrect, need at-least two items in nodeLst.")
        //For each w_ij in WMatx, 0 < i < d^{l-1} ; 1 < j < d^{l}
        //biases for each node in layer l will be added as extra node in layer l-1 hence i>=0
        //i rows ; j cols
        
    | _ -> Error("The specification was incorrect, nodeLst must be exactly one item greater than actFnLst.")


let fwdProp (x: Vector<float>) (network : Result<Network, ErrType>) : Result<(NodesVectList * (Layer list)), ErrType> =
    // take in a vector of type floa this is our input can be mx1 for any m dim of input
    // also takes in a network that will get reduced
    // finally returns a vector which will be our hypothesis h(x) can be of any dim matching y dim
    //actFn((w.T * x + b))

    let propNetwork (nvectLstAndLayers: NodesVectList * (Layer list)) (layer : Layer) : (NodesVectList * (Layer list)) =
        
        let nvectLst, layerLst = nvectLstAndLayers
         //result of matmul will be nx1 matx where n=l^(d) so excluding the "1" node or x_0^l(d)
        let newSVect =  layer.wMatx.Transpose() * CreateVector.Dense((Array.concat [ [|1.0|] ; nvectLst.Head.xVect.ToArray() ]))  //bias is auto included by weight[0, :]
        let newXVect = (newSVect.Map (Func<float, float> (getActFn layer.actFn)))   //Func type-casting needed by vector
        ({xVect=newXVect ; sVect=newSVect} :: nvectLst , layer :: layerLst)

        //for list.mapfold u give X^(0) as state_0 and Weight_(1) then result X_(1) is new state and also used for mapping so you return it twice
        //next time u give X^(1) as state_1 and Weight_(2) to get X_(2)
        //in end u have list of X's and final state == X_(L)

    match network with
    | Error(e) -> Error(e) //pass-through any prev errors
    | Ok(net) ->
        (List.fold propNetwork ([{xVect=x ; sVect=x}], []) net.lLst)   //initially sVect == xVect i.e. no actFn for raw input
        //|> fst  //List.mapFold will return NodesVect list * NodesVect, we only need the first item
        //|> List.rev
        |> (fun x -> Ok(x))

let backProp (yVect : Vector<float>) (fwdPropEval : Result<(NodesVectList * (Layer list)), ErrType>) : Result<Layer list, ErrType> =
    //functions
    (*
    
    
    *)

    let calcPrevDelta (deltaLst : Vector<float> list) (layer: Layer) (nodesInLayer: NodesVect) =
        //printfn "***WMATX --> \n\n%A\n\ndeltaLstzDELTA_CURR --> \n\n%A\n\n" layer deltaLst
        printfn "****DEBUG PREVDELTA: %A\n ***DEBUG wMatx: %A, \nDEBUG: nodesVect: %A" deltaLst.Head layer.wMatx nodesInLayer

        

        let interim = layer.wMatx.RemoveRow(0) * deltaLst.Head

        let theta' = nodesInLayer.sVect.Map (Func<float, float> (getActFnDeriv (layer.actFn)))
        printfn "***INTERIM --> \n\n%A\n\nTHETA' --> \n\n%A\n\n" interim theta'

        //hamadard product
        let prevDelta = interim.PointwiseMultiply(theta')
        prevDelta :: deltaLst

    

    match fwdPropEval with
    | Error(e) -> Error(e)
    | Ok((xAndSLstRev, netRev)) ->
        //temporary for debug dont use noise
        //let revList = List.rev xAndSLst
        //let tupleLst = List.map2 (fun xAndS layer -> (xAndS, layer)) xAndSLst net.lLst
        
        let len = xAndSLstRev.Length

        let deltaL = lastLayerDeriv xAndSLstRev.Head.xVect xAndSLstRev.Head.sVect yVect netRev.Head.actFn
        printfn "***DEBUG_FINAL_LAYER_ERR_DERIV: %A" deltaL
        // first step is to calculate last layer error
        // this uses a combination of derivative of mean square error and derivative of last layer act Fn
        

        let rec _calcPrevDelta (deltaLst : Vector<float> list) (layerLst: Layer list) (nodesInLayerLst: NodesVect list) : Vector<float> list =
            match layerLst with
            | [head] -> deltaLst
            | head :: head2 :: tail -> 
                        printfn "****DEBUG PREVDELTA: %A\n ***DEBUG wMatx: %A, \nDEBUG: nodesVect: %A" deltaLst.Head head.wMatx nodesInLayerLst
                        let interim = head.wMatx.RemoveRow(0) * deltaLst.Head

                        let theta' = nodesInLayerLst.Head.sVect.Map (Func<float, float> (getActFnDeriv (head2.actFn)))
                        printfn "***INTERIM --> \n\n%A\n\nTHETA' --> \n\n%A\n\n" interim theta'

                        let layerLst = 
                            match layerLst with
                            | _ :: tail -> tail
                            | [] -> [] //throw error??
                        let nodesInLayerLst = 
                            match nodesInLayerLst with
                            | _ :: tail -> tail
                            | [] -> [] //throw error??
                        //hamadard product
                        let prevDelta = interim.PointwiseMultiply(theta')
                        _calcPrevDelta (prevDelta :: deltaLst) (layerLst) (nodesInLayerLst)
            | [] -> []//throw error []

        let subRevLst = 
            match xAndSLstRev with
            | _ :: tail -> tail
            | [] -> [] //throw error??
        //so u foldback and calc dleta and every time you get new delta u append to head so that is O(1)
        //A list of all error deltas in reverse order
        //first arg must be a list
        //slice the xAndSList cause we need the prev elements for multiply
        let deltaLst = _calcPrevDelta [deltaL] netRev subRevLst //List.fold2 calcPrevDelta [deltaL] netRev subRevLst 

        printfn "THE DELTAS: \n\n%A" deltaLst

        Ok(netRev)


//define data
let XActual = [ 1.0 .. 1.0 .. 10.0 ]
let XTrain = [ 1.0 .. 1.0 .. 10.0 ]


let plotResults finalResult = 
    use pl = new PLStream()
    PLplot.Native.sdev("wincairo")
    pl.init()
    pl.env( 0.0, 10.0, 0.0, 30.0, AxesScale.Independent, AxisBox.BoxTicksLabelsAxes )
    
    //entry point
    // match finalResult with
    // | (finalMSE, finalW) ->
    //     pl.lab( "x ->", "y ->", (sprintf "A plot of true values ('x') vs final prediction (line) [MSE: %0.4f]" finalMSE))
    //     pl.poin( List.toArray(XTrain), List.toArray(getYList yLookupMap), 'x')
    //     pl.line( List.toArray(XTrain), List.toArray(List.map (fun h -> (h finalW)) H))

[<EntryPoint>]
let main argv = 
    //optimise weights, print final results and then plot the results

    printfn "ID(-343434.0) == %A" ((getActFn ID) -343434.0)

    // printfn "%A" "test"
    // (initNetwork [ 1 ; 3 ; 1] [ID ; ID] ID')
    // |> printfn "%A"

    //***real code
    //let nnArch = (initNetwork [ 1 ; 3 ; 1] [TANH ; ID] ID')

    let x = 0.2
    let y = yRealFn x //temp for now we only do without noise for debugging

    printfn "Y_VAL****: %A" y

    let w_l1 = DenseMatrix.OfRowArrays([| [|0.1 ; 0.15|] ; [|0.05 ; 0.2|] |])
    let w_l2 = DenseMatrix.OfRowArrays([| [|0.7|] ; [|-0.6|] ; [|0.4|] |])
    let debugNNArch = {
        lLst = [ {wMatx=w_l1 ; actFn=TANH} ; {wMatx=w_l2 ; actFn=ID} ]
    }

    printfn "***DEBUG_ARCH: %A\n" debugNNArch

    let debugfWDPropRes =
        Ok(debugNNArch)
        |> fwdProp (CreateVector.Dense([|x|]))
        |> function
           | Ok(x) -> fst x |> printfn "***DEBUG_FWD-PROP_RESULT: \n\n%A"

    let debugBackPropRes =
        Ok(debugNNArch)
        |> fwdProp (CreateVector.Dense([|x|]))
        |> backProp (CreateVector.Dense([|y|]))
        |> function
           | Ok(x) -> x //|> //printfn "***DEBUG_FWD-PROP_RESULT: \n\n%A"


    //***Real code
    // let fwdPropResult = 
    //     nnArch
    //     |> fwdProp (CreateVector.Dense([|1.0|]))
    // match fwdPropResult with
    // | Ok(x) -> fst x |> printfn "FWD-PROP RESULT: \n\n%A"
    // | Error(x) -> x |> printfn "FWD-PROP ERROR: \n\n%A"
     
    //printfn "RESULT: \n\n%A" tt
    // (optimise XTrain H yLookupMap wMain 1)  
    // |> (fun x -> (printfn "\n\n(finalMSEAndWeights, finalWeights): %A" x) ; x)
    // |> plotResults

    0 // return an integer exit code
