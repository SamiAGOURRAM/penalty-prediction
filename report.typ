#set page(paper: "a4", margin: (x: 1.8cm, y: 1.8cm), numbering: "1")
#set text(size: 10pt, font: "Libertinus Serif")
#set par(justify: true, leading: 0.62em)
#let navy = rgb("#2b3a67")
#let teal = rgb("#1f9e89")
#show heading: set text(fill: navy)
#show heading.where(level: 1): it => block(above: 1.0em, below: 0.5em, text(size: 13pt, it))
#show heading.where(level: 2): it => block(above: 0.75em, below: 0.3em, text(size: 10.5pt, it))
#let kpi(v, l) = align(center, box[#text(size: 14pt, fill: teal, weight: "bold")[#v] \ #text(size: 8pt, fill: gray)[#l]])

#align(center)[
  #text(size: 17pt, weight: "bold", fill: navy)[Where Will the Penalty Go?]
  #v(-6pt)
  #text(size: 10.5pt, style: "italic")[Kicker placement and goalkeeper dives in 1,300+ penalties from public data]
]

#block(fill: rgb("#eef2f7"), inset: 9pt, radius: 4pt, stroke: (left: 3pt + navy))[
  *Abstract.* A penalty is an asymmetric guessing game between kicker and goalkeeper. Using 1,032
  open-play and shootout penalties (StatsBomb) and 320 shootout kicks with coded dives (Kaggle),
  we report three results. First, kicker direction is close to unpredictable beyond footedness, so
  the value of a probabilistic model lies in calibration rather than top-1 accuracy. Second,
  goalkeepers show a strong static directional bias but no measurable sequential dependence. Third,
  combining the two implies a modest, largely unexploited scoring edge from shooting to the
  non-natural side. Each result is reported with explicit uncertainty.
]

#grid(columns: 5, gutter: 6pt,
  kpi("1,032", "StatsBomb penalties"), kpi("320", "shootout kicks with dives"),
  kpi("88%", "of keepers commit to a side"), kpi("0.78 vs 0.71", "P(score): non-natural vs natural"),
  kpi("4 / 4", "sequential tests null"))

= Data and methods

*Data.* The kicker model uses StatsBomb open data (1,032 penalties, 534 kickers, 14 competitions;
placement L/C/R from the shot end-location). The goalkeeper analysis uses the Kaggle World Cup
shootouts, 1982 to 2022 (35 shootouts, 320 kicks), because StatsBomb does not record dive
direction: the keeper is logged at the set point on the goal line, which we confirmed on the raw
events including saved penalties.

*Coding.* The two corpora label left and right from opposite viewpoints. Placement and dive are
therefore recoded as *natural*, *centre*, or *non-natural* relative to the kicker's foot, which is
frame-invariant and the only consistent basis for combining the sources.

*Models and tests.* Kicker direction is modelled with a Bayesian hierarchical multinomial logit
(a footedness baseline, a shootout indicator, and a shooter random intercept for partial pooling),
sampled with nutpie (4 chains, 2,000 draws; maximum R-hat 1.005). It is compared against the
footedness marginal, a per-player frequency table, and a logit without random effects, and scored
on log-loss, Brier score, and expected calibration error (ECE), with an exact-binomial McNemar
test, forward-chaining temporal cross-validation, and a sensitivity sweep over the centre-band
width. Goalkeeper sequential dependence is tested with an exact binomial test, a permutation test,
and bootstrap intervals, with explicit statistical power. The placement-by-dive payoff table and
the expected-score comparison use a cluster bootstrap over shootouts, and reported proportions
carry Wilson 95% intervals.

= 1. Kicker direction: footedness dominates, and the model's value is calibration

At the population level, placement is largely determined by footedness: right-footers favour one
side and left-footers the mirror image, with an overall split of 50% natural, 14% centre, 36%
non-natural. Beyond footedness, direction is close to random, and top-1 accuracy is bounded near
50% because takers mix near the game-theoretic equilibrium. A model therefore cannot reliably
out-guess the kicker; its value lies in well-calibrated probabilities, in the sense that
predictions assigned 80% probability should occur about 80% of the time.

#grid(columns: (1fr, 1fr), gutter: 8pt,
  figure(image("outputs/figs/fig_calibration.png", width: 100%),
    caption: [Reliability of each predictor. The per-player frequency table (red) is markedly
    over-confident on low-sample takers; the hierarchical model and the footedness marginal lie
    near the diagonal. ECE in parentheses.]),
  figure(image("outputs/figs/fig_shrinkage.png", width: 100%),
    caption: [Raw versus partially pooled natural-side estimate. The model trusts a player's record
    in proportion to its size: Messi (83 penalties) retains his lean; Candreva (8) is shrunk toward
    the population mean.])
)

The hierarchical model is well calibrated (ECE 0.02) and clearly outperforms the per-player table
(ECE 0.12), which is badly over-confident on the many takers with few recorded penalties. It also
attains lower out-of-sample log-loss than the table at both sample-size extremes (low-n 1.01 vs
1.11; high-n 0.94 vs 1.02; exact McNemar #emph[p] = 0.047). The mechanism is partial pooling: each
player's estimate is drawn toward his own record only in proportion to the evidence behind it.
Individual tendencies are real but small: the shooter random-effect scale is non-zero (posterior
about 0.4, 94% interval excluding zero), so placement is mostly footedness plus a modest personal
lean that is well identified only for high-volume takers. One contextual effect is robust: under
shootout pressure the natural-side share rises from 49% to 55%, a shift in the placement
distribution that is distinct from the documented effects of pressure on conversion.

= 2. Goalkeeper behaviour: a strong static bias, no sequential signal

Goalkeepers are far more predictable than kickers, but only in aggregate. They commit to a side on
88% of kicks and lean toward the kicker's natural side. They show no usable sequential structure:
across four hypotheses, the gambler's fallacy (diving opposite a streak), dependence on the
previous kick, autocorrelation between successive dives, and win-stay/lose-shift on the previous
outcome, every effect is null (all #emph[p] > 0.4). The lag-1 test is adequately powered (176
cases; 99% power against a true 65% tendency) and returns 51%, so this is evidence of absence
rather than absence of evidence, which addresses the long-standing dispute over the gambler's
fallacy in shootouts on modern data.

#figure(image("outputs/figs/fig_keeper.png", width: 92%),
  caption: [Left: the goalkeeper dive distribution (relative to the kicker's natural side). Right:
  the rate of diving opposite after a same-direction kick streak, with bootstrap 95% intervals; all
  intervals include 0.5, and the lag-1 case is well powered.])

The exploitable quantity is therefore the static directional bias, not any in-shootout pattern.

= 3. Exploiting the goalkeeper's directional bias

Because keepers over-commit to the natural side, the placement-by-dive payoff table is asymmetric:
the keeper saves chiefly when his dive matches the kick, leaving the non-natural side comparatively
under-defended. Expected score is highest for the non-natural corner (0.78, against 0.71 natural
and 0.66 centre), and the realized rates agree (0.75, 0.67, 0.64).

#figure(image("outputs/figs/fig_bridge.png", width: 100%),
  caption: [Left: P(goal) by kick placement and keeper dive; the keeper saves mainly on the
  diagonal. Right: expected score (bars) and realized score (points) by placement, against the
  always-natural baseline.])

This corresponds to roughly eight additional goals per hundred relative to always shooting the
natural side, consistent with the long-noted but previously unquantified observation that kickers
under-exploit goalkeeper bias. The estimate is limited by sample size: with 35 shootouts the 95%
interval on the gain runs from about three goals below zero to eighteen above, so the effect is
directionally consistent but not statistically conclusive, and it is a counterfactual that holds
goalkeeper behaviour fixed.

= 4. Why the centre is not the high-value option

A natural objection is that, since keepers commit to a side 88% of the time, the centre should be
open. The data do not support this, and the relevant comparison is expected value, which weights
each conversion by how often it occurs.

Conditional on the keeper diving away, a central shot scores 0.70 [0.57, 0.80] while a non-natural
corner scores 0.96 [0.88, 0.99]: a diving keeper's legs still cover the centre, so vacating a side
is not the same as opening the goal. Weighting by frequency, a central kick meets an absent keeper
88% of the time yet converts only 0.70, and meets a stationary keeper 12% of the time, converting
0.25 [0.07, 0.59] (on eight kicks, hence imprecise). The expected score is therefore
$0.88 times 0.70 + 0.12 times 0.25 approx 0.66$, below the non-natural corner's 0.78. Even
conditional on being on target, the centre converts worst (0.72 against 0.79). The genuinely
under-defended region is the non-natural corner, which the keeper reaches on only 46% of such
kicks and cannot reach on 96% of his mistakes.

#figure(image("outputs/figs/fig_middle.png", width: 78%),
  caption: [P(goal) by placement, split by whether the keeper guessed the side (teal: guessed
  wrong; grey: guessed right), with Wilson 95% intervals. The wide centre/right interval reflects
  only eight kicks.])

= 5. Practical implications

== For the penalty taker

Ranked by expected score: non-natural corner (0.78), natural corner (0.71), centre (0.66). The
keeper occupies the natural side on 61% of natural-side kicks but only 46% of non-natural kicks,
and when he guesses wrong a corner is near-certain whereas the centre is not. Priorities follow:
(i) a reliably on-target corner struck with pace is the highest-value skill, since corners convert
only when on frame and the non-natural corner is on target 95% of the time; (ii) competence with
both corners, because a one-sided taker is readily scouted and the keeper already favours the
natural side; (iii) the centre as a low-frequency, disguised option only, given its poor
conversion when anticipated. Less skilled strikers should prioritise an on-target corner over the
centre, while stronger strikers can place a corner beyond the reach of a correct dive. Against a
goalkeeper of exceptional reach the central and mid-height regions shrink fastest, further
favouring low, wide placement (a reasoned extension, not measured here).

#figure(image("outputs/figs/fig_execution.png", width: 100%),
  caption: [Left: P(goal | shot on target) by placement; corners convert better than the centre
  even conditional on hitting the target. Right: the keeper match rate, P(keeper on the kicked
  side). Each bar is a separate conditional rate, so they need not sum to one.])

== For the goalkeeper

Within a shootout there is no sequence to exploit, so preparation should target a specific taker's
placement distribution rather than the order of kicks. Once committed, guessing the correct side
is the sole determinant of a save: a matched dive saves about half of corner kicks, a mismatched
dive almost none. Central shots are stopped by the legs even on a committed dive, so the centre
should not be abandoned. At the population level goalkeepers may be too predictable: a keeper who
always commits, and always to the natural side, is precisely the pattern the kicker analysis
exploits, so occasional central holds or non-obvious choices raise the cost of preparing against
him. Training that delays commitment or extends reach converts incorrect guesses into partial saves.

== For the analyst and coaching staff

+ *Data.* Placement is available in StatsBomb open data; dive direction is not (the keeper is
  logged on his line), so video coding, a commercial feed, or a set such as the Kaggle shootouts
  is required. Code from the goalkeeper's viewpoint and fix the centre-band width in advance.
+ *Conventions.* Reconcile left/right viewpoints before merging sources, and work in natural,
  centre, and non-natural relative to the foot.
+ *Shrinkage.* Do not use raw per-player frequencies: most takers have fewer than ten penalties on
  record and raw counts are over-confident. Pool toward the footedness prior and attach a
  confidence weight; trust a strong individual read only for high-volume takers.
+ *Scope.* Sequential and momentum models are uninformative for both agents here; prioritise
  placement distributions and reachability.
+ *Application.* For the taker, identify the opposition keeper's directional lean and target the
  corner he vacates; for the keeper, supply the taker's pooled placement distribution, noting that
  most takers are near-random and the edge is small.
+ *Uncertainty.* Report intervals and flag thin cells; the "keeper holds the centre" save rate
  here rests on eight kicks and should be treated as indicative only.

= Limitations

The goalkeeper analyses rest on 35 shootouts, so several cells are thin. The headline centre
figures (0.64 overall, 0.70 conditional on the keeper diving away) sit on 56 to 64 kicks and are
reasonably precise, but the "holds the centre, saves three in four" figure rests on eight kicks
and is indicative only. Selection is also present, since takers who choose the centre are not a
random sample. The kicker corpus is mildly concentrated (Messi accounts for 9% of penalties), but
the placement split is unchanged when high-volume takers are removed, and the Kaggle-based tactical
results contain no player identities and survive leave-one-team-out resampling, so neither result
is driven by a single player. The exploitation gain is a counterfactual that holds keeper behaviour
fixed and its interval still includes zero. Finally, StatsBomb logs no dive direction, so the
goalkeeper side cannot be enlarged without additional data.

= References

#set text(size: 8.5pt)
- Misirlisoy, E. and Haggard, P. (2014). Asymmetric predictability and cognitive competition in football penalty shootouts. _Current Biology_ 24(16), 1918 to 1922.
- Braun, S. and Schmidt, U. (2015). The gambler's fallacy in penalty shootouts (with reply by Misirlisoy and Haggard). _Current Biology_ 25(14).
- Bar-Eli, M., Azar, O. H., Ritov, I., Keidar-Levin, Y. and Schein, G. (2007). Action bias among elite soccer goalkeepers: the case of penalty kicks. _Journal of Economic Psychology_ 28(5), 606 to 621.
- Palacios-Huerta, I. (2003). Professionals play minimax. _Review of Economic Studies_ 70(2), 395 to 415.
- Chiappori, P.-A., Groseclose, T. and Levitt, S. (2002). Testing mixed-strategy equilibria when players are heterogeneous: the case of penalty kicks in soccer. _American Economic Review_ 92(4), 1138 to 1151.
- Tea, P. and Swartz, T. (2023). Bayesian hierarchical modelling of serve direction in tennis. _Annals of Operations Research_.
- Jordet, G., Hartman, E., Visscher, C. and Lemmink, K. (2007). Kicks from the penalty mark in soccer: stress, skill and fatigue. _Journal of Sports Sciences_ 25(2), 121 to 129.
- Data: StatsBomb Open Data (github.com/statsbomb/open-data); Kaggle, World Cup Penalty Shootouts 1982 to 2022.

#align(center, text(size: 8pt, fill: gray)[
  Every number is reproducible from the scripts in `scripts/`.
])
