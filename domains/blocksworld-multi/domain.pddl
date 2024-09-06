(define (domain blocksworld-4ops)
  (:requirements :strips)
   (:types robot object)
(:predicates (clear ?x - object)
             (on-table ?x - object)
             (arm-empty ?r - robot)
             (holding ?r - robot ?x - object)
             (on ?x - object ?y - object))

(:action pickup
  :parameters (?r - robot ?ob - object)
  :precondition (and (clear ?ob) (on-table ?ob) (arm-empty ?r))
  :effect (and (holding ?r ?ob) (not (clear ?ob)) (not (on-table ?ob)) 
               (not (arm-empty ?r))))

(:action putdown
  :parameters  (?r - robot ?ob - object)
  :precondition (holding ?r ?ob)
  :effect (and (clear ?ob) (arm-empty ?r) (on-table ?ob) 
               (not (holding ?r ?ob))))

(:action stack
  :parameters  (?r - robot ?ob - object ?underob - object)
  :precondition (and (clear ?underob) (holding ?r ?ob))
  :effect (and (arm-empty ?r) (clear ?ob) (on ?ob ?underob)
               (not (clear ?underob)) (not (holding ?r ?ob))))

(:action unstack
  :parameters  (?r - robot ?ob - object ?underob - object)
  :precondition (and (on ?ob ?underob) (clear ?ob) (arm-empty ?r))
  :effect (and (holding ?r ?ob) (clear ?underob)
               (not (on ?ob ?underob)) (not (clear ?ob)) (not (arm-empty ?r)))))
