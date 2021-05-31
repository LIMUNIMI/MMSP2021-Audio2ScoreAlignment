# in module DTW...

"""
A Component contains two `Int64` and a weight `Float64`. These represent the
quantity subtracted from the position (i, j), e.g.:

    `weight * d[i - row, j - col]`
"""
struct Component
    row::Int64 # row index difference with the current position
    col::Int64 # col index difference with the current position
    weight::Float64
end

function _consistency_check(components::Vector{Component})::Vector{Component}
    # last component must be [0, 0]
    if components[end].row != 0 || components[end].row != 0
        error("Last components of rules must be [0, 0]")
    end

    # components in decreasing rows and cols
    prev_row = components[1].row
    prev_col = components[1].col
    @views for comp in components[2:end]
        if comp.row > prev_row
            error("Rules must have rows in decreasing order")
        end
        if comp.col > prev_col
            error("Rules must have cols in decreasing order")
        end
    end

    return components
end

"""
A Rule contains a `Vector{Component}` so that:

1. the last component should always be 0, 0
2. both rows and cols should be in decreasing order

The components are summed so that a rule is created
"""
struct Rule
    # the steps starting from the end
    components::Vector{Component}
    weight::Float64
    bias::Float64
    # a consistency check for rules
    Rule(x) = new(_consistency_check(x), 1.0, 0)
    Rule(x; weight=1.0, bias=0.0) = new(_consistency_check(x), weight, bias)
    Rule(x, weight, bias) = new(_consistency_check(x), weight, bias)
end

"""
A StepPattern contains a `Vector{Rule}` sorted from the most important to the least important

You can print it by using `recursion_rule`
"""
struct StepPattern
    rules::Vector{Rule}
    norm_hint::String
    weight::Float64

    # a constructor with no normalization hint and no fede weight
    StepPattern(x) = new(x, "NA", 1.0)

    # a constructor for dtw-python objects with normalization hint
    StepPattern(x, y::String) = new(x, y, 1.0)

    # a constructor for fede objects with weight
    StepPattern(x, y::Float64) = new(x, "NA", y)

    # a constructor for fede objects with weight and normalization hint
    StepPattern(x, y, z) = new(x, y, z)
end

"""
Returns a new same step-pattern with row and cols inverted (transposed)
"""
function tr(x::StepPattern)::StepPattern
    rules = Vector{Rule}(undef, length(x.rules))
    for (r, rule) in enumerate(x.rules)
        components = Vector{Component}(undef, length(rule.components))
        for (c, comp) in enumerate(rule.components)
            components[c] = Component(comp.col, comp.row, comp.weight)
        end
        rules[r] = Rule(components)
    end
    return StepPattern(rules)
end

"""
A function to convert dtw-python step-patterns
"""
function _c(args...)
    num_rule = 0
    row = col = weight = 0
    rules = Rule[]
    components = Component[]
    for (i, arg) in enumerate(args)
        el = i % 4
        if el == 1
            # a new component
            # push back the previous component if any
            if num_rule > 0
                push!(components, Component(row, col, weight))
            end

            if arg != num_rule
                # a new rule
                # push back the previous rule if any
                if num_rule > 0
                    push!(rules, Rule(components))
                end

                # prepare a new vector of components for this new rule
                components = Component[]
                num_rule = arg
            end
        elseif el == 2
            # the row of this component
            row = arg
        elseif el == 3
            # the col of this component
            col = arg
        elseif el == 0
            # the weight of this component
            weight = abs(arg)
        end
    end
    # push back last component
    push!(components, Component(row, col, weight))
    # push back last rule
    push!(rules, Rule(components))
    return rules
end

#! format: off
"""
an asymmetric pattern which favours the vertical paths (changing row is
easier than changing column)
"""
const fede_asymmetric = StepPattern(
                             [
                              Rule([
                                    Component(1, 1, 1),
                                    Component(0, 0, 3),
                                   ]),

                              Rule([
                                    Component(1, 0, 1),
                                    Component(0, 0, 1),
                                   ]),

                              Rule([
                                    Component(0, 1, 1),
                                    Component(0, 0, 2),
                                   ]),

                              Rule([
                                    Component(2, 1, 1),
                                    Component(1, 0, 2),
                                    Component(0, 0, 1),
                                   ]),

                              Rule([
                                    Component(2, 1, 1),
                                    Component(1, 1, 2),
                                    Component(0, 0, 1),
                                   ]),

                              Rule([
                                    Component(2, 1, 1),
                                    Component(0, 0, 3),
                                   ]),

                              Rule([
                                    Component(1, 2, 1),
                                    Component(0, 1, 2),
                                    Component(0, 0, 2),
                                   ]),

                              Rule([
                                    Component(1, 2, 1),
                                    Component(1, 1, 2),
                                    Component(0, 0, 2),
                                   ]),

                              Rule([
                                    Component(1, 2, 1),
                                    Component(0, 0, 4),
                                   ]),
                             ]
                            )

"""
a symmetric pattern for DTW
"""
const fede_symmetric = StepPattern(
                        [
                              Rule([
                                    Component(1, 1, 1),
                                    Component(0, 0, 3),
                                   ]),

                              Rule([
                                    Component(1, 0, 1),
                                    Component(0, 0, 2),
                                   ]),

                              Rule([
                                    Component(0, 1, 1),
                                    Component(0, 0, 2),
                                   ]),

                              Rule([
                                    Component(2, 1, 1),
                                    Component(1, 0, 2),
                                    Component(0, 0, 2),
                                   ]),

                              Rule([
                                    Component(2, 1, 1),
                                    Component(1, 1, 2),
                                    Component(0, 0, 2),
                                   ]),

                              Rule([
                                    Component(2, 1, 1),
                                    Component(0, 0, 4),
                                   ]),

                              Rule([
                                    Component(1, 2, 1),
                                    Component(0, 1, 2),
                                    Component(0, 0, 2),
                                   ]),

                              Rule([
                                    Component(1, 2, 1),
                                    Component(1, 1, 2),
                                    Component(0, 0, 2),
                                   ]),

                              Rule([
                                    Component(1, 2, 1),
                                    Component(0, 0, 4),
                                   ]),
                        ]
                       )

"""
Given a vector of weights creates a step-pattern using standard rules
"""
function weight_step_pattern(w::AbstractVector{Float64})::StepPattern
    return StepPattern([
        Rule([
                Component(1, 1, w[1]),
                Component(0, 0, w[2])
            ],
            w[3], w[4] # weight and bias
        ),

        Rule([
                Component(1, 0, w[5]),
                Component(0, 0, w[6])
            ],
            w[7], w[8] # weight and bias
        ),

        Rule([
                Component(0, 1, w[9]),
                Component(0, 0, w[10])
            ],
            w[11], w[12] # weight and bias
        )],
        w[13]
      )

    # return StepPattern([
    #     Rule([
    #           Component(1, 1, w[1]),
    #           Component(0, 0, w[2]),
    #           ],
    #          1, 0),
    #          # w[3], w[4]),

    #     Rule([
    #           Component(1, 0, w[3]),
    #           Component(0, 0, w[4]),
    #           ],
    #          1, 0),
    #          # w[7], w[8],

    #     Rule([
    #           Component(0, 1, w[5]),
    #           Component(0, 0, w[6]),
    #           ],
    #          1, 0),
    #          # w[11], w[12]),

    #     Rule([
    #           Component(2, 1, w[7]),
    #           Component(1, 0, w[8]),
    #           Component(0, 0, w[9]),
    #           ],
    #          1, 0),
    #          # w[16], w[17]),

    #     Rule([
    #           Component(2, 1, w[10]),
    #           Component(1, 1, w[11]),
    #           Component(0, 0, w[12]),
    #           ],
    #          1, 0),
    #          # w[21], w[22]),

    #     Rule([
    #           Component(2, 1, w[13]),
    #           Component(0, 0, w[14]),
    #           ],
    #          1, 0),
    #          # w[25], w[26]),

    #     Rule([
    #           Component(1, 2, w[15]),
    #           Component(0, 1, w[16]),
    #           Component(0, 0, w[17]),
    #           ],
    #          1, 0),
    #          # w[30], w[31]),

    #     Rule([
    #           Component(1, 2, w[18]),
    #           Component(1, 1, w[19]),
    #           Component(0, 0, w[20]),
    #           ],
    #          1, 0),
    #          # w[35], w[36]),

    #     Rule([
    #           Component(1, 2, w[21]),
    #           Component(0, 0, w[22]),
    #          ],
    #          1, 0),
    #          # w[39], w[40]),
    #    ],
    #    w[23])
end

##########################################################################################
##########################################################################################

## Everything here is semi auto-generated from the python source, which is
## auto-generated from the R source. Don't edit!

##################################################
##################################################


##
## Various step patterns, defined as internal variables
##
## First column: enumerates step patterns.
## Second   	 step in query index
## Third	 step in reference index
## Fourth	 weight if positive, or -1 if starting point
##
## For cite{} see dtw.bib in the package
##


## Widely-known variants

"""
White-Neely symmetric (default)
aka Quasi-symmetric cite{White1976}
normalization: no (N+M?)
"""
const symmetric1 = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 1,
    2, 0, 1, -1,
    2, 0, 0, 1,
    3, 1, 0, -1,
    3, 0, 0, 1
))

"""
Normal symmetric
normalization: N+M
"""
const symmetric2 = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 2,
    2, 0, 1, -1,
    2, 0, 0, 1,
    3, 1, 0, -1,
    3, 0, 0, 1
), "N+M")

"""
classic asymmetric pattern: max slope 2, min slope 0
normalization: N
"""
const asymmetric = StepPattern(_c(
    1, 1, 0, -1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 0, 1
), "N")

# % item{code{symmetricVelichkoZagoruyko}}{symmetric, reproduced from %
# [Sakoe1978]. Use distance matrix code{1-d}}
#

"""
normalization: max[N,M]
note: local distance matrix is 1-d
cite{Velichko}
"""
const _symmetricVelichkoZagoruyko = StepPattern(_c(
    1, 0, 1, -1,
    1, 0, 0, 0,
    2, 1, 1, -1,
    2, 0, 0, -1.001,
    3, 1, 0, -1,
    3, 0, 0, 0,
    ))

# % item{code{asymmetricItakura}}{asymmetric, slope contrained 0.5 -- 2
# from reference [Itakura1975]. This is the recursive definition % that
# generates the Itakura parallelogram }
#

"""
Itakura slope-limited asymmetric cite{Itakura1975}
Max slope: 2 min slope: 1/2
normalization: N
"""
const _asymmetricItakura = StepPattern(_c(
    1, 1, 2, -1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 2, 1, -1,
    3, 1, 0, 1,
    3, 0, 0, 1,
    4, 2, 2, -1,
    4, 1, 0, 1,
    4, 0, 0, 1
))

#############################
## Slope-limited versions
##
## Taken from Table I, page 47 of "Dynamic programming algorithm
## optimization for spoken word recognition," Acoustics, Speech, and
## Signal Processing, vol.26, no.1, pp. 43-49, Feb 1978 URL:
## http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1163055
##
## Mostly unchecked


"""
Row P=0
"""
symmetricP0 = symmetric2


"""
normalization: N ?
"""
const asymmetricP0 = StepPattern(_c(
    1, 0, 1, -1,
    1, 0, 0, 0,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 0, -1,
    3, 0, 0, 1
), "N")

"""
alternative implementation
"""
const _asymmetricP0b = StepPattern(_c(
    1, 0, 1, -1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 0, -1,
    3, 0, 0, 1
), "N")

"""
Row P=1/2
"""
const symmetricP05 = StepPattern(_c(
    1, 1, 3, -1,
    1, 0, 2, 2,
    1, 0, 1, 1,
    1, 0, 0, 1,
    2, 1, 2, -1,
    2, 0, 1, 2,
    2, 0, 0, 1,
    3, 1, 1, -1,
    3, 0, 0, 2,
    4, 2, 1, -1,
    4, 1, 0, 2,
    4, 0, 0, 1,
    5, 3, 1, -1,
    5, 2, 0, 2,
    5, 1, 0, 1,
    5, 0, 0, 1
), "N+M")

const asymmetricP05 = StepPattern(_c(
    1, 1, 3, -1,
    1, 0, 2, 1 / 3,
    1, 0, 1, 1 / 3,
    1, 0, 0, 1 / 3,
    2, 1, 2, -1,
    2, 0, 1, .5,
    2, 0, 0, .5,
    3, 1, 1, -1,
    3, 0, 0, 1,
    4, 2, 1, -1,
    4, 1, 0, 1,
    4, 0, 0, 1,
    5, 3, 1, -1,
    5, 2, 0, 1,
    5, 1, 0, 1,
    5, 0, 0, 1
), "N")

"""
Row P=1
Implementation of Sakoe's P=1, Symmetric algorithm
"""
const symmetricP1 = StepPattern(_c(
    1, 1, 2, -1,  # First branch: g(i-1,j-2)+
    1, 0, 1, 2,  # + 2d(i  ,j-1)
    1, 0, 0, 1,  # +  d(i  ,j)
    2, 1, 1, -1,  # Second branch: g(i-1,j-1)+
    2, 0, 0, 2,  # +2d(i,  j)
    3, 2, 1, -1,  # Third branch: g(i-2,j-1)+
    3, 1, 0, 2,  # + 2d(i-1,j)
    3, 0, 0, 1  # +  d(  i,j)
), "N+M")

const asymmetricP1 = StepPattern(_c(
    1, 1, 2, -1,
    1, 0, 1, .5,
    1, 0, 0, .5,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 2, 1, -1,
    3, 1, 0, 1,
    3, 0, 0, 1
), "N")

"""
Row P=2
"""
const symmetricP2 = StepPattern(_c(
    1, 2, 3, -1,
    1, 1, 2, 2,
    1, 0, 1, 2,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 2,
    3, 3, 2, -1,
    3, 2, 1, 2,
    3, 1, 0, 2,
    3, 0, 0, 1
), "N+M")

const asymmetricP2 = StepPattern(_c(
    1, 2, 3, -1,
    1, 1, 2, 2 / 3,
    1, 0, 1, 2 / 3,
    1, 0, 0, 2 / 3,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 3, 2, -1,
    3, 2, 1, 1,
    3, 1, 0, 1,
    3, 0, 0, 1
), "N")

################################
## Taken from Table III, page 49.
## Four varieties of DP-algorithm compared

## 1st row:  asymmetric

## 2nd row:  symmetricVelichkoZagoruyko

## 3rd row:  symmetric1

## 4th row:  asymmetricItakura


#############################
## Classified according to Rabiner
##
## Taken from chapter 2, Myers' thesis [4]. Letter is
## the weighting function:
##
##      rule       norm   unbiased
##   a  min step   ~N     NO
##   b  max step   ~N     NO
##   c  x step     N      YES
##   d  x+y step   N+M    YES
##
## Mostly unchecked

# R-Myers     R-Juang
# type I      type II
# type II     type III
# type III    type IV
# type IV     type VII


const typeIa = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 1,
    1, 0, 0, 0,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 1, 1,
    3, 0, 0, 0
))

const typeIb = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 1, 1,
    3, 0, 0, 1
))

const typeIc = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 1, 1,
    3, 0, 0, 0
), "N")

const typeId = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 2,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 2,
    3, 1, 2, -1,
    3, 0, 1, 2,
    3, 0, 0, 1
), "N+M")

## ----------
## smoothed variants of above

const typeIas = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, .5,
    1, 0, 0, .5,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 1, .5,
    3, 0, 0, .5
))

const typeIbs = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 1, 1,
    3, 0, 0, 1
))

const typeIcs = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 1, 2, -1,
    3, 0, 1, .5,
    3, 0, 0, .5
), "N")

const typeIds = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 1.5,
    1, 0, 0, 1.5,
    2, 1, 1, -1,
    2, 0, 0, 2,
    3, 1, 2, -1,
    3, 0, 1, 1.5,
    3, 0, 0, 1.5
), "N+M")

## ----------

const typeIIa = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 1,
    2, 1, 2, -1,
    2, 0, 0, 1,
    3, 2, 1, -1,
    3, 0, 0, 1
))

const typeIIb = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 1,
    2, 1, 2, -1,
    2, 0, 0, 2,
    3, 2, 1, -1,
    3, 0, 0, 2
))

const typeIIc = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 1,
    2, 1, 2, -1,
    2, 0, 0, 1,
    3, 2, 1, -1,
    3, 0, 0, 2
), "N")

const typeIId = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 2,
    2, 1, 2, -1,
    2, 0, 0, 3,
    3, 2, 1, -1,
    3, 0, 0, 3
), "N+M")

## ----------

## Rabiner [3] discusses why this is not equivalent to Itakura's

const typeIIIc = StepPattern(_c(
    1, 1, 2, -1,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 1,
    3, 2, 1, -1,
    3, 1, 0, 1,
    3, 0, 0, 1,
    4, 2, 2, -1,
    4, 1, 0, 1,
    4, 0, 0, 1
), "N")

## ----------

## numbers follow as production rules in fig 2.16

const typeIVc = StepPattern(_c(
    1, 1, 1, -1,
    1, 0, 0, 1,
    2, 1, 2, -1,
    2, 0, 0, 1,
    3, 1, 3, -1,
    3, 0, 0, 1,
    4, 2, 1, -1,
    4, 1, 0, 1,
    4, 0, 0, 1,
    5, 2, 2, -1,
    5, 1, 0, 1,
    5, 0, 0, 1,
    6, 2, 3, -1,
    6, 1, 0, 1,
    6, 0, 0, 1,
    7, 3, 1, -1,
    7, 2, 0, 1,
    7, 1, 0, 1,
    7, 0, 0, 1,
    8, 3, 2, -1,
    8, 2, 0, 1,
    8, 1, 0, 1,
    8, 0, 0, 1,
    9, 3, 3, -1,
    9, 2, 0, 1,
    9, 1, 0, 1,
    9, 0, 0, 1
), "N")

#############################
##
"""
Mori's asymmetric step-constrained pattern. Normalized in the
reference length.

Mori, A. Uchida, S. Kurazume, R. Taniguchi, R. Hasegawa, T. &
Sakoe, H. Early Recognition and Prediction of Gestures Proc. 18th
International Conference on Pattern Recognition ICPR 2006, 2006, 3,
560-563
"""
const mori2006 = StepPattern(_c(
    1, 2, 1, -1,
    1, 1, 0, 2,
    1, 0, 0, 1,
    2, 1, 1, -1,
    2, 0, 0, 3,
    3, 1, 2, -1,
    3, 0, 1, 3,
    3, 0, 0, 3
), "M")

"""
Completely unflexible: fixed slope 1. Only makes sense with
open.begin and open.end

N.B. open begin and open end are not implemented yet in Julia
"""
const rigid = StepPattern(_c(1, 1, 1, -1,
                       1, 0, 0, 1), "N")
##########################################################################################
##########################################################################################


function get_default_step_patterns()
    return (
        # fede symmetric pattern
        fede_symmetric,

        # fede asymmetric pattern
        fede_asymmetric,
        tr(fede_asymmetric),

        # dtw-python symmetric pattern
        symmetric1,
        symmetric2,
        symmetricP05,
        symmetricP1,
        symmetricP2,
        _symmetricVelichkoZagoruyko,

        # dtw-python asymmetric pattern
        asymmetric,
        tr(asymmetric),
        asymmetricP0,
        tr(asymmetricP0),
        asymmetricP05,
        tr(asymmetricP05),
        asymmetricP1,
        tr(asymmetricP1),
        asymmetricP2,
        tr(asymmetricP2),
        mori2006,
        tr(mori2006),
        _asymmetricItakura,
        tr(_asymmetricItakura),
    )
end
#! format: on

"""
Just prints the recursion rule from a pattern
"""
function recursion_rule(step_pattern::StepPattern)
    println("d[i, j] = min(")
    for rule in step_pattern.rules
        print("\t$(rule.weight) * [")
        for (i, comp) in enumerate(rule.components)
            print("$(comp.weight) * d[i-$(comp.row), j-$(comp.col)]")
            if i < length(rule.components)
                print(" + ")
            else
                print("] + $(rule.bias),\n")
            end
        end
    end
    return println(")")
end
