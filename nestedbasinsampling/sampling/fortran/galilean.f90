
module galilean

integer(kind=4), save :: seed=1

contains

function r_uniform() result(rand)

!*****************************************************************************80
!
!! R8_UNIFORM_01 returns a unit pseudorandom R8.
!
!  Discussion:
!
!    This routine implements the recursion
!
!      seed = 16807 * seed mod ( 2^31 - 1 )
!      r8_uniform_01 = seed / ( 2^31 - 1 )
!
!    The integer arithmetic never requires more than 32 bits,
!    including a sign bit.
!
!    If the initial seed is 12345, then the first three computations are
!
!      Input     Output      R8_UNIFORM_01
!      SEED      SEED
!
!         12345   207482415  0.096616
!     207482415  1790989824  0.833995
!    1790989824  2035175616  0.947702
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    31 May 2007
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Paul Bratley, Bennett Fox, Linus Schrage,
!    A Guide to Simulation,
!    Second Edition,
!    Springer, 1987,
!    ISBN: 0387964673,
!    LC: QA76.9.C65.B73.
!
!    Bennett Fox,
!    Algorithm 647:
!    Implementation and Relative Efficiency of Quasirandom
!    Sequence Generators,
!    ACM Transactions on Mathematical Software,
!    Volume 12, Number 4, December 1986, pages 362-376.
!
!    Pierre L'Ecuyer,
!    Random Number Generation,
!    in Handbook of Simulation,
!    edited by Jerry Banks,
!    Wiley, 1998,
!    ISBN: 0471134031,
!    LC: T57.62.H37.
!
!    Peter Lewis, Allen Goodman, James Miller,
!    A Pseudo-Random Number Generator for the System/360,
!    IBM Systems Journal,
!    Volume 8, 1969, pages 136-143.
!
!  Parameters:
!
!    Input/output, integer ( kind = 4 ) SEED, the "seed" value, which
!    should NOT be 0.
!    On output, SEED has been updated.
!
!    Output, real ( kind = 8 ) R8_UNIFORM_01, a new pseudorandom variate,
!    strictly between 0 and 1.
!
  implicit none

  integer ( kind = 8 ) k
  real ( kind = 8 ) rand
!  integer ( kind = 4 ) seed

  k = seed / 127773

  seed = 16807 * ( seed - k * 127773 ) - k * 2836

  if ( seed < 0 ) then
    seed = seed + 2147483647
  end if
!
!  Although SEED can be represented exactly as a 32 bit integer,
!  it generally cannot be represented exactly as a 32 bit real number!
!
  rand = real ( seed, kind = 8 ) * 4.656612875D-10

  return
end

function r_normal () result(rand)

!*****************************************************************************80
!
!! R8_NORMAL_01 returns a unit pseudonormal R8.
!
!  Discussion:
!
!    The standard normal probability distribution function (PDF) has
!    mean 0 and standard deviation 1.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    06 August 2013
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input/output, integer ( kind = 4 ) SEED, a seed for the random
!    number generator.
!
!    Output, real ( kind = 8 ) R8_NORMAL_01, a normally distributed
!    random value.
!
  implicit none

  real ( kind = 8 ) r1
  real ( kind = 8 ) r2
  real ( kind = 8 ) rand
  real ( kind = 8 ), parameter :: r8_pi = 3.141592653589793D+00
 ! real ( kind = 8 ) r8_uniform_01
 ! integer ( kind = 4 ) seed
  real ( kind = 8 ) x

  r1 = r_uniform ()
  r2 = r_uniform ()
  rand = sqrt ( - 2.0D+00 * log ( r1 ) ) * cos ( 2.0D+00 * r8_pi * r2 )

  return
end

function random_unitvec(n)

implicit none

integer n
double precision, dimension(n) :: random_unitvec

integer i
double precision rand, d2

do i=1,n
    rand = r_normal()
    random_unitvec(i) = rand
    d2 = d2 + rand**2
end do

random_unitvec(:) = random_unitvec(:)/sqrt(d2)

end

subroutine newpoint(ecut, coords, n, nsteps, maxreject, stepsize, &
    & pot, constraint, energy, grad, endcoords, &
    & naccept, nreject, nreflect, niter, info)

implicit none

integer, intent(in) :: n, nsteps, maxreject
double precision, intent(in) :: ecut, stepsize, coords(n)
double precision, intent(out) :: energy, grad(n), endcoords(n)
integer, intent(out) :: naccept, nreject, nreflect, niter, info

external pot
!f2py intent(in), depend(n) :: testcoords
!f2py intent(hide) :: n
!f2py intent(out) :: enew
!f2py intent(out), depend(n) :: gnew
external constraint
!f2py intent(in), depend(n) :: testcoords
!f2py intent(hide) :: n
!f2py intent(out) :: econ
!f2py intent(out), depend(n) :: gcon

double precision gnew(n), gcon(n), p(n), newcoords(n), testcoords(n)
double precision gn(n), gn1(n), gn2(n), d1, d2
double precision econ, enew, eold

integer j

logical eaccept, caccept

nreject = 0
naccept = 0
nreflect = 0
niter = 0

! testing whether the starting configuration is a valid starting point
testcoords(:) = coords(:)

call pot(testcoords, enew, gnew, n)
call constraint(testcoords, econ, gcon, n)

caccept = econ.le.0.d0
eaccept = enew.le.ecut

if (.not.eaccept.or.(.not.caccept)) then
    info = 2
    return
end if

eold = enew
newcoords(:) = coords
p(:) = random_unitvec(n) * stepsize

do while(niter.lt.nsteps)

    testcoords(:) = newcoords(:) + p(:)

    call pot(testcoords, enew, gnew, n)
    call constraint(testcoords, econ, gcon, n)

    caccept = econ.le.0.d0
    eaccept = enew.le.ecut

    if (eaccept.and.caccept) then
        newcoords(:) = testcoords(:)
        naccept = naccept + 1
        niter = niter + 1
    else

        if (eaccept.and.(.not.caccept)) then
            d2 = 0.d0
            do j=1,n
                d2 = d2 + gcon(j)**2
            end do
            gn(:) = gcon(:)/sqrt(d2)
        else if(caccept.and.(.not.eaccept)) then
            d2 = 0.d0
            do j=1,n
                d2 = d2 + gnew(j)**2
            end do
            gn(:) = gnew(:)/sqrt(d2)
        else
            d1 = 0.d0
            d2 = 0.d0
            do j=1,n
                d1 = d1 + gnew(j)**2
                d2 = d2 + gcon(j)**2
            end do
            gn(:) = gnew(:)/sqrt(d1) + gcon(:)/sqrt(d2)
            d2 = 0.d0
            do j=1,n
                d2 = d2 + gn(j)
            end do
            gn(:) = gn(:)/sqrt(d2)
        end if
        p(:) = p(:) - 2.d0 * gn(:) * dot_product(gn, p)
        testcoords(:) = testcoords(:) + p(:)

        call pot(testcoords, enew, gnew, n)
        call constraint(testcoords, econ, gcon, n)

        caccept = econ.le.0.d0
        eaccept = enew.le.ecut

        if (eaccept.and.caccept) then
            newcoords(:) = testcoords(:)
            niter = niter + 1
            nreflect = nreflect + 1
        else
            enew = eold
            niter = niter + 1
            nreject = nreject + 1
        end if

        if (nreject.gt.maxreject) then
            info = 3
            return
        end if

        p(:) = random_unitvec(n) * stepsize
    end if
end do

info = 1
energy = enew
grad = gnew
endcoords(:) = newcoords(:)

end subroutine newpoint


end module galilean
