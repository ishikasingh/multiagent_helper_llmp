

(define (problem BW-rand-3)
(:domain blocksworld-4ops)
(:objects robot1 robot2 - robot
    b1 b2 b3 - object )
(:init
(arm-empty robot1)
(arm-empty robot2)
(on-table b1)
(on b2 b3)
(on b3 b1)
(clear b2)
)
(:goal
(and
(on b3 b2)
(on b1 b3))
)
)


