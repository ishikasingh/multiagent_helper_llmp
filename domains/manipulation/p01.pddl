(define (problem manipulation-01)
(:domain manipulation)
(:objects
coffee-table side-table recycle-bin pantry - location
mustard-bottle soup-can - object)
(:init
(= (total-cost) 0)
(= (distance coffee-table coffee-table) 0)
(= (distance side-table side-table) 0)
(= (distance recycle-bin recycle-bin) 0)
(= (distance pantry pantry) 0)
(= (distance coffee-table side-table) 10)
(= (distance side-table coffee-table) 10)
(= (distance coffee-table recycle-bin) 10)
(= (distance recycle-bin coffee-table) 10)
(= (distance coffee-table pantry) 20)
(= (distance pantry coffee-table) 20)
(= (distance side-table recycle-bin) 1)
(= (distance recycle-bin side-table) 1)
(= (distance side-table pantry) 10)
(= (distance pantry side-table) 10)
(= (distance recycle-bin pantry) 10)
(= (distance pantry recycle-bin) 10)
(robot-at coffee-table)
(at mustard-bottle coffee-table)
(at soup-can side-table)
(hand-empty)
)
(:goal
(and
(at mustard-bottle pantry)
(at soup-can recycle-bin)
)
)
(:metric minimize (total-cost))
)