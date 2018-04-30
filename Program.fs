// Learn more about F# at http://fsharp.org
//XPlot.Plotly.WPF

open System
open PLplot
open MathNet.Numerics.Distributions
open MathNet.Numerics.Random
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double

type ErrType = string

type Weight = {w1: float; w2: float; b: float}



type ActFn = | ID    //identity f(x) = x
             | TANH // f(x) = tanh(x)
              //| RELU

type Layer = {wMatx: Matrix<float>; actFn: ActFn;}

type Network = Layer list

type NodesVect = {sVect: Vector<float>; xVect: Vector<float>} //each nodesVect will be S => without actFn and X => with actFn

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

let epochs = 1//50
let lr = 0.04//0.0001

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

//Only works with MSE as metrics!
let lastLayerDeriv (x_L : Vector<float>) (s_L : Vector<float>) (y : Vector<float>) (theta_L' : ActFn) =
    (2.0*(x_L - y)).PointwiseMultiply(s_L.Map (Func<float, float> (getActFnDeriv (theta_L'))))
    



//provide start, step, end and yFn
//return a tuple list with each element as (x_i, y_i)
let genDataSet (startVal) (endVal) (stepVal) (fN: 'a -> 'b) = 
    [ startVal .. stepVal .. endVal]
    |> List.map(fun x -> (x, yRealFnAndNoise x))
    


let initNetwork (nodeLst : int list) (actFnLst : ActFn list) : Result<Network, ErrType> =
    //or ([14 ; 32 ; 8 ; 1], [relu, relu, identity]) so list2 must be 1 less or else return result.error
    //return is [ (wmtx1, actFnLayer1) ... (wmatxL, actFnLayerL) ]
    let genRandMatx (rows) (cols) : Matrix<float> = 
        CreateMatrix.Random(rows,cols)

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
            |> Ok
        | _ -> Error("The specification was incorrect, need at-least two items in nodeLst.")
        //For each w_ij in WMatx, 0 < i < d^{l-1} ; 1 < j < d^{l}
        //biases for each node in layer l will be added as extra node in layer l-1 hence i>=0
        //i rows ; j cols
        
    | _ -> Error("The specification was incorrect, nodeLst must be exactly one item greater than actFnLst.")


let fwdProp (x: Vector<float>) (net : Network) : (NodesVectList * (Network)) =
    // take in a vector of type floa this is our input can be mx1 for any m dim of input
    // also takes in a network that will get reduced
    // finally returns a vector which will be our hypothesis h(x) can be of any dim matching y dim
    //actFn((w.T * x + b))

    let propNetwork (nvectLstAndLayers: NodesVectList * (Network)) (layer : Layer) : (NodesVectList * (Network)) =
        
        let nvectLst, layerLst = nvectLstAndLayers
         //result of matmul will be nx1 matx where n=l^(d) so excluding the "1" node or x_0^l(d)
        let newSVect =  layer.wMatx.Transpose() * CreateVector.Dense((Array.concat [ [|1.0|] ; nvectLst.Head.xVect.ToArray() ]))  //bias is auto included by weight[0, :]
        let newXVect = (newSVect.Map (Func<float, float> (getActFn layer.actFn)))   //Func type-casting needed by vector
        ({xVect=newXVect ; sVect=newSVect} :: nvectLst , layer :: layerLst)

        //for list.mapfold u give X^(0) as state_0 and Weight_(1) then result X_(1) is new state and also used for mapping so you return it twice
        //next time u give X^(1) as state_1 and Weight_(2) to get X_(2)
        //in end u have list of X's and final state == X_(L)

    (List.fold propNetwork ([{xVect=x ; sVect=x}], []) net)   //initially sVect == xVect i.e. no actFn for raw input

let backProp (yVect : Vector<float>) (fwdPropEval : (NodesVectList * (Network))) : Network =

    let rec _calcPrevDelta (deltaLst : Vector<float> list) (layerLst: Network) (nodesInLayerLst: NodesVect list) : Vector<float> list =
        match layerLst with
        | [_] -> deltaLst
        | head :: head2 :: _ -> 
                    printfn "****DEBUG PREVDELTA: %A\n ***DEBUG wMatx: %A, \nDEBUG: nodesVect: %A" deltaLst.Head head.wMatx nodesInLayerLst
                    let interim = head.wMatx.RemoveRow(0) * deltaLst.Head

                    let theta' = nodesInLayerLst.Head.sVect.Map (Func<float, float> (getActFnDeriv (head2.actFn)))
                    printfn "***INTERIM --> \n\n%A\n\nTHETA' --> \n\n%A\n\n" interim theta'

                    //hamadard product
                    let prevDelta = interim.PointwiseMultiply(theta')
                    _calcPrevDelta (prevDelta :: deltaLst) (layerLst.Tail) (nodesInLayerLst.Tail)
        | [] -> [] //should throw error but this wil never hit if wMatx is checked initially
    
    let layerListUpdater (prevNodes : NodesVect) (currDelta : Vector<float>) (currLayer : Layer) : Layer =
        //must add 1.0 on top for correct shape!
        //torow-matx is same as (colVect).T
        
        let dEdWMatx =  CreateMatrix.DenseOfColumnArrays( (Array.concat [ [|1.0|] ; prevNodes.xVect.ToArray() ])  ) * currDelta.ToRowMatrix()

        printfn "dEdWMATXXXXX: \n%A\n" dEdWMatx
        // printfn "dEdWMATXXXXX Reduced: \n%A\n" (lr*dEdWMatx)

        let newWMatx = currLayer.wMatx - (lr*dEdWMatx)

        printfn "***updatedW: \n%A\n" newWMatx
        {currLayer with wMatx=newWMatx}
        

    let xAndSLstRev = fwdPropEval |> fst
    let netRev = fwdPropEval |> snd

    //temporary for debug dont use noise

    let deltaL = lastLayerDeriv xAndSLstRev.Head.xVect xAndSLstRev.Head.sVect yVect netRev.Head.actFn
    printfn "***DEBUG_FINAL_LAYER_ERR_DERIV: %A" deltaL
    // first step is to calculate last layer error
    // this uses a combination of derivative of mean square error and derivative of last layer act Fn

    //so u foldback and calc dleta and every time you get new delta u append to head so that is O(1)
    //A list of all error deltas in reverse order
    //first arg must be a list
    //slice the xAndSList cause we need the prev elements for multiply
    let deltaRevLst = _calcPrevDelta [deltaL] netRev xAndSLstRev.Tail |> List.rev//List.fold2 calcPrevDelta [deltaL] netRev subRevLst 

    //now simply use list fold2 to calc Wmatx from deltaRevLst and xAndSRevLst.Tail
    //also since they r rev if for every wMatx you append to front, then in the end your new wMAtxList will be automatically sorted!!
    printfn "THE DELTAS: \n\n%A" deltaRevLst
    List.map3 layerListUpdater xAndSLstRev.Tail deltaRevLst netRev |> List.rev
        



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
    match (initNetwork [ 1 ] [ ]) with
    | Ok(x) -> printfn "OK: %A" x
    | Error(x) -> printfn "ERROR: %A" x

    let x = 0.2
    let y = yRealFn x //temp for now we only do without noise for debugging

    printfn "Y_VAL****: %A" y

    let w_l1 = DenseMatrix.OfRowArrays([| [|0.1 ; 0.15|] ; [|0.05 ; 0.2|] |])
    let w_l2 = DenseMatrix.OfRowArrays([| [|0.7|] ; [|-0.6|] ; [|0.4|] |])
    let debugNNArch = [ {wMatx=w_l1 ; actFn=TANH} ; {wMatx=w_l2 ; actFn=ID} ]

    printfn "***DEBUG_ARCH: %A\n" debugNNArch

    debugNNArch
    |> fwdProp (CreateVector.Dense([|x|]))
    |> backProp (CreateVector.Dense([|y|]))
    |> printfn "***DEBUG_BACK-PROP_RESULT: \n\n%A" //|> //printfn "***DEBUG_FWD-PROP_RESULT: \n\n%A"


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
