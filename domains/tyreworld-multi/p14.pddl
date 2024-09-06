(define (problem tyreworld-14)
(:domain tyreworld)
(:objects 
wrench jack pump - tool
the-hub1 the-hub2 the-hub3 the-hub4 the-hub5 the-hub6 the-hub7 the-hub8 the-hub9 the-hub10 the-hub11 the-hub12 the-hub13 the-hub14 - hub
nuts1 nuts2 nuts3 nuts4 nuts5 nuts6 nuts7 nuts8 nuts9 nuts10 nuts11 nuts12 nuts13 nuts14 - nut
boot - container
r1 w1 r2 w2 r3 w3 r4 w4 r5 w5 r6 w6 r7 w7 r8 w8 r9 w9 r10 w10 r11 w11 r12 w12 r13 w13 r14 w14 - wheel
robot1 robot2 - robot
)
(:init
(in jack boot)
(in pump boot)
(in wrench boot)
(unlocked boot)
(closed boot)
(intact r1)
(in r1 boot)
(not-inflated r1)
(intact r2)
(in r2 boot)
(not-inflated r2)
(intact r3)
(in r3 boot)
(not-inflated r3)
(intact r4)
(in r4 boot)
(not-inflated r4)
(intact r5)
(in r5 boot)
(not-inflated r5)
(intact r6)
(in r6 boot)
(not-inflated r6)
(intact r7)
(in r7 boot)
(not-inflated r7)
(intact r8)
(in r8 boot)
(not-inflated r8)
(intact r9)
(in r9 boot)
(not-inflated r9)
(intact r10)
(in r10 boot)
(not-inflated r10)
(intact r11)
(in r11 boot)
(not-inflated r11)
(intact r12)
(in r12 boot)
(not-inflated r12)
(intact r13)
(in r13 boot)
(not-inflated r13)
(intact r14)
(in r14 boot)
(not-inflated r14)
(on w1 the-hub1)
(on-ground the-hub1)
(tight nuts1 the-hub1)
(fastened the-hub1)
(on w2 the-hub2)
(on-ground the-hub2)
(tight nuts2 the-hub2)
(fastened the-hub2)
(on w3 the-hub3)
(on-ground the-hub3)
(tight nuts3 the-hub3)
(fastened the-hub3)
(on w4 the-hub4)
(on-ground the-hub4)
(tight nuts4 the-hub4)
(fastened the-hub4)
(on w5 the-hub5)
(on-ground the-hub5)
(tight nuts5 the-hub5)
(fastened the-hub5)
(on w6 the-hub6)
(on-ground the-hub6)
(tight nuts6 the-hub6)
(fastened the-hub6)
(on w7 the-hub7)
(on-ground the-hub7)
(tight nuts7 the-hub7)
(fastened the-hub7)
(on w8 the-hub8)
(on-ground the-hub8)
(tight nuts8 the-hub8)
(fastened the-hub8)
(on w9 the-hub9)
(on-ground the-hub9)
(tight nuts9 the-hub9)
(fastened the-hub9)
(on w10 the-hub10)
(on-ground the-hub10)
(tight nuts10 the-hub10)
(fastened the-hub10)
(on w11 the-hub11)
(on-ground the-hub11)
(tight nuts11 the-hub11)
(fastened the-hub11)
(on w12 the-hub12)
(on-ground the-hub12)
(tight nuts12 the-hub12)
(fastened the-hub12)
(on w13 the-hub13)
(on-ground the-hub13)
(tight nuts13 the-hub13)
(fastened the-hub13)
(on w14 the-hub14)
(on-ground the-hub14)
(tight nuts14 the-hub14)
(fastened the-hub14)
)
(:goal
(and
(on r1 the-hub1)
(inflated r1)
(tight nuts1 the-hub1)
(in w1 boot)
(on r2 the-hub2)
(inflated r2)
(tight nuts2 the-hub2)
(in w2 boot)
(on r3 the-hub3)
(inflated r3)
(tight nuts3 the-hub3)
(in w3 boot)
(on r4 the-hub4)
(inflated r4)
(tight nuts4 the-hub4)
(in w4 boot)
(on r5 the-hub5)
(inflated r5)
(tight nuts5 the-hub5)
(in w5 boot)
(on r6 the-hub6)
(inflated r6)
(tight nuts6 the-hub6)
(in w6 boot)
(on r7 the-hub7)
(inflated r7)
(tight nuts7 the-hub7)
(in w7 boot)
(on r8 the-hub8)
(inflated r8)
(tight nuts8 the-hub8)
(in w8 boot)
(on r9 the-hub9)
(inflated r9)
(tight nuts9 the-hub9)
(in w9 boot)
(on r10 the-hub10)
(inflated r10)
(tight nuts10 the-hub10)
(in w10 boot)
(on r11 the-hub11)
(inflated r11)
(tight nuts11 the-hub11)
(in w11 boot)
(on r12 the-hub12)
(inflated r12)
(tight nuts12 the-hub12)
(in w12 boot)
(on r13 the-hub13)
(inflated r13)
(tight nuts13 the-hub13)
(in w13 boot)
(on r14 the-hub14)
(inflated r14)
(tight nuts14 the-hub14)
(in w14 boot)
(in wrench boot)
(in jack boot)
(in pump boot)
(closed boot)
)
)
)
