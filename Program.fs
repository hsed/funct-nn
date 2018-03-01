// Learn more about F# at http://fsharp.org
//XPlot.Plotly.WPF

open System
open PLplot
open MathNet.Numerics.Distributions
open MathNet.Numerics.Random

type Weight = {w1: float; w2: float; b: float}

let yRealFn x = 0.4*(x**float(2)) + 0.3*x + 0.9 + Normal.Sample(0.0, 1.0)

//hypothesis
let hypFn (x: float) (w:Weight) =
            w.w1*(x**float(2)) + w.w2*x + w.b

let costFnWithPow (pow) (h : float) (y: float) =
            (h - y)**pow

// the actual costFn
let costFn = costFnWithPow(2.0)

//partial differntiation
//dJ/dw1, dJ/dw2, dJ/db
let getCostDerivatives x h y : (float * float * float) = 
                        let costTerm = (costFnWithPow 1.0 h y)
                        let m = 2.0 //multiplier
                        (
                            m*(x**2.0)*costTerm,
                            m*(x)*costTerm,
                            m*costTerm
                        )


let transformList (yFunc) (X) : (float list * float list)=
    X
    |> (fun X -> (X, (List.map yFunc X)))

let rng = SystemRandomSource.Default

[<EntryPoint>]
let main argv = 
    
    let wMain = {
        w1 = rng.NextDouble();
        w2 = rng.NextDouble();
        b = rng.NextDouble();
    }
   
    use pl = new PLStream()
    pl.init()
    pl.env( 0.0, 10.0, 0.0, 30.0, AxesScale.Independent, AxisBox.BoxTicksLabelsAxes )
    pl.lab( "x", "y=100 x#u2#d", "Simple PLplot demo of a 2D line plot" )
    
    let XActual = [ 1.0 .. 1.0 .. 10.0 ]
    let XTrain = [ 1.0 .. 1.0 .. 10.0 ]
    //this is a map now
    // it will always be >= the hypothesis ranges
    let yLookupMap : Map<float,float> = 
        XActual
        |> List.map (fun x -> (x, yRealFn(x)))
        |> Map.ofList
        
    let H =
        XTrain
        |> List.map (fun x -> hypFn(x))

    printfn "X: %A\n\nH:%A" XTrain H
    //H = a vector with each elem f(w1,w2,b) but not x
    //Y = a vector with each elem const
    let lr = 0.0001

    let rec optimise (xList : float list) (hList: (Weight -> float) list) (yMap : Map<float,float>) (w : Weight) (i : int) =
        let getDerivTuple : ((float * float * float) list) =
            (List.map2 (fun x h -> (getCostDerivatives x (h w) (yMap.[x])) ) xList hList)
        
        let getCosts : (float list) =
            (List.map2 (fun x h -> (costFn (h w) (yMap.[x])) ) xList hList)

        printfn "FinalDerivTupleList: %A" getDerivTuple
        printfn "FinalCostList: %A" getCosts

        let avgWError (tripleTuple : (float *float *float) list) : (float * float * float) =
            let m = float(tripleTuple.Length)
            ((List.sumBy (fun (w1, _, _) -> w1) tripleTuple) / m,
             (List.sumBy (fun (_, w2, _) -> w2) tripleTuple) / m,
             (List.sumBy (fun (_, _, b) -> b) tripleTuple) / m    )

        let MSE = List.average getCosts

        printfn "FinalAvgError: %A" (avgWError getDerivTuple)

        let updateW wOrig w1E w2E bE lr =
            {
                w1 = wOrig.w1 - lr*w1E
                w2 = wOrig.w2 - lr*w2E
                b  = wOrig.b  - lr*bE 
            }

        match i with
        | index when (index < 50) ->
            match (avgWError getDerivTuple) with
            | (w1E,w2E,bE) -> optimise xList hList yLookupMap (updateW w w1E w2E bE lr) (i+1)                  
        |  _ -> (MSE, w)       


    let finalMSE = (optimise XTrain H yLookupMap wMain 1)
    
    let getYList map = 
        map
        |> Map.toList
        |> List.map snd
        |> List.distinct

    printfn "finalMSE!!!: %A" finalMSE 
    //Chart.Line [ for x in 1.0 .. 5.0 -> (x, yRealFn(x)) ] |> ignore
    pl.poin( List.toArray(XTrain), List.toArray(getYList yLookupMap), 'x')
    match finalMSE with
    | (_, finalW) -> pl.line( List.toArray(XTrain), List.toArray(List.map (fun h -> (h finalW)) H))

    

    

    0 // return an integer exit code
