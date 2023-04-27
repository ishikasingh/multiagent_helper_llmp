

(define (problem painting-problem)
  (:domain painting)
  (:objects
    tile_0-1 tile_0-2 tile_0-3 tile_0-4 tile_0-5
    tile_1-1 tile_1-2 tile_1-3 tile_1-4 tile_1-5
    tile_2-1 tile_2-2 tile_2-3 tile_2-4 tile_2-5
    tile_3-1 tile_3-2 tile_3-3 tile_3-4 tile_3-5
    tile_4-1 tile_4-2 tile_4-3 tile_4-4 tile_4-5
    tile_5-1 tile_5-2 tile_5-3 tile_5-4 tile_5-5
    robot1 robot2
    white black
  )
  (:init
    (at robot1 tile_5-3)
    (at robot2 tile_1-5)
    (color white tile_1-1)
    (color black tile_1-2)
    (color white tile_1-3)
    (color black tile_1-4)
    (color white tile_1-5)
    (color black tile_2-1)
    (color white tile_2-2)
    (color black tile_2-3)
    (color white tile_2-4)
    (color black tile_2-5)
    (color white tile_3-1)
    (color black tile_3-2)
    (color white tile_3-3)
    (color black tile_3-4)
    (color white tile_3-5)
    (color black tile_4-1)
    (color white tile_4-2)
    (color black tile_4-3)
    (color white tile_4-4)
    (color black tile_4-5)
    (color white tile_5-1)
    (color black tile_5-2)
    (color white tile_5-3)
    (color black tile_5-4)
    (color white tile_5-5)
    (color-gun robot1 white)
    (color-gun robot2 black)
  )
  (:goal (and
    (painted white tile_1-1)
    (painted black tile_1-2)
    (painted white tile_1-3)
    (painted black tile_1-4)
    (painted white tile_1-5)
    (painted black tile_2-1)
    (painted white tile_2-2)
    (painted black tile_2-3)
    (painted white tile_2-4)
    (painted black tile_2-5)
    (painted white tile_3-1)
    (painted black tile_3-2)
    (painted white tile_3-3)
    (painted black tile_3-4)
    (painted white tile_3-5)
    (painted black tile_4-1)
    (painted white tile_4-2)
    (painted black tile_4-3)
    (painted white tile_4-4)
    (painted black tile_4-5)
    (painted white tile_5-1)
    (painted black tile_5-2)
    (painted white tile_5-3)
    (painted black tile_5-4)
    (painted white tile_5-5)
  ))
  (:metric minimize (total-cost))
)