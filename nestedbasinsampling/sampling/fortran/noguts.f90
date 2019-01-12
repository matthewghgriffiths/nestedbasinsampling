module noguts

integer, parameter  :: int32  = selected_int_kind(9)
integer, parameter  :: int64  = selected_int_kind(18)
integer, parameter  :: real32 = selected_real_kind(6, 37)
integer, parameter  :: real64 = selected_real_kind(15, 307)

integer(int32), save :: seed=1
logical, save :: remove_linear_momentum=.FALSE., remove_angular_momentum=.FALSE.
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

  integer(int32) k
  double precision rand
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
  rand = dble(seed) * 4.656612875D-10

  return
end

function r_normal ()

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

  double precision r1
  double precision r2
  double precision r_normal
  double precision, parameter :: r8_pi = 3.141592653589793D+00
 ! real ( kind = 8 ) r8_uniform_01
 ! integer ( kind = 4 ) seed

  r1 = r_uniform ()
  r2 = r_uniform ()
  r_normal = sqrt ( - 2.0D+00 * log ( r1 ) ) * cos ( 2.0D+00 * r8_pi * r2 )

  return
end

function random_normal_vec(n)

implicit none

integer n
double precision, dimension(n) :: random_normal_vec

integer i

do i=1,n
    random_normal_vec(i) = r_normal()
end do

end

function random_unitvec(n)

implicit none

integer n
double precision, dimension(n) :: random_unitvec

integer i
double precision d2

random_unitvec(:) = random_normal_vec(n)

d2 = 0.d0
do i=1,n
    d2 = d2 + random_unitvec(i)**2
end do
random_unitvec(:) = random_unitvec(:)/sqrt(d2)

end

subroutine takestep( &
    & coords, p_f, p_b, v, Ecut, epsilon, E, G, accept, pot, constraint, n)
implicit none

integer, intent(in) :: n, v
double precision, intent(inout) :: coords(n), p_f(n), p_b(n)
double precision, intent(in) :: Ecut, epsilon
double precision, intent(out) :: E, G(n)
logical, intent(out) :: accept

external pot
!f2py intent(in), depend(n) :: coords
!f2py intent(hide) :: n
!f2py intent(out) :: E
!f2py intent(out), depend(n) :: G
external constraint
!f2py intent(in), depend(n) :: coords
!f2py intent(hide) :: n
!f2py intent(out) :: D
!f2py intent(out), depend(n) :: F

integer k
double precision D, F(n), L2, M2
logical Eaccept, Caccept

if (v.eq.-1) then
    coords(1:n) = coords(1:n) - p_b(1:n) * epsilon
    call pot(coords, E, G, n)
    call constraint(coords, D, F, n)
    Eaccept = Ecut.gt.E
    Caccept = 0.d0.gt.D
    if (Eaccept.and.Caccept) then
        ! do nothing
        accept = .TRUE.
    else if (Eaccept.and.(.not.Caccept)) then
        ! reflect off constraint gradient
        accept = .FALSE.
        L2 = 0.d0
        do k=1,n
            L2 = L2 + F(k)**2
        end do
        p_f(1:n) = p_b(1:n)
        p_b(1:n) = p_b(1:n) - 2.d0 * F(1:n) * dot_product(F(1:n), p_b(1:n)) / L2
    else if(Caccept.and.(.not.Eaccept)) then
        ! reflect off gradient
        accept = .FALSE.
        do k=1,n
            L2 = L2 + G(k)**2
        end do
        p_f(1:n) = p_b(1:n)
        p_b(1:n) = p_b(1:n) - 2.d0 * G(1:n) * dot_product(G(1:n), p_b(1:n)) / L2
    else if((.not.Eaccept).and.(.not.Caccept)) then
        ! reflect of sum of normed gradient and constraint gradient
        accept = .FALSE.
        L2 = 0.d0
        M2 = 0.d0
        do k=1,n
            L2 = L2 + G(k)**2
            M2 = M2 + F(k)**2
        end do
        L2 = sqrt(L2)
        M2 = sqrt(M2)
        D = 0.d0
        do k=1,n
            F(k) = F(k)/M2 + G(k)/L2
            D = D + F(k)**2
        end do
        p_f(1:n) = p_b(1:n)
        p_b(1:n) = p_b(1:n) - 2.d0 * F(1:n) * dot_product(F(1:n), p_b(1:n)) / D
    end if
else
    ! Assuming v==1
    coords(1:n) = coords(1:n) + p_f(1:n) * epsilon
    call pot(coords, E, G, n)
    call constraint(coords, D, F, n)
    Eaccept = Ecut.gt.E
    Caccept = 0.d0.gt.D
    if (Eaccept.and.Caccept) then
        accept = .TRUE.
    else if (Eaccept.and.(.not.Caccept)) then
        ! reflect off constraint gradient
        accept = .FALSE.
        L2 = 0.d0
        do k=1,n
            L2 = L2 + F(k)**2
        end do
        p_b(1:n) = p_f(1:n)
        p_f(1:n) = p_f(1:n) - 2.d0 * F(1:n) * dot_product(F(1:n), p_f(1:n)) / L2
    else if(Caccept.and.(.not.Eaccept)) then
        ! reflect off gradient
        accept = .FALSE.
        L2 = 0.d0
        do k=1,n
            L2 = L2 + G(k)**2
        end do
        p_b(1:n) = p_f(1:n)
        p_f(1:n) = p_f(1:n) - 2.d0 * G(1:n) * dot_product(G(1:n), p_f(1:n)) / L2
    else if((.not.Eaccept).and.(.not.Caccept)) then
        ! reflect of sum of normed gradient and constraint gradient
        accept = .FALSE.
        L2 = 0.d0
        M2 = 0.d0
        do k=1,n
            L2 = L2 + G(k)**2
            M2 = M2 + F(k)**2
        end do
        L2 = sqrt(L2)
        M2 = sqrt(M2)
        D = 0.d0
        do k=1,n
            F(k) = F(k)/M2 + G(k)/L2
            D = D + F(k)**2
        end do
        p_b(1:n) = p_f(1:n)
        p_f(1:n) = p_f(1:n) - 2.d0 * F(1:n) * dot_product(F(1:n), p_f(1:n)) / D
    end if
end if

end subroutine

recursive subroutine build_tree( &
    & X_pls, p_pls_f, p_pls_b, X_min, p_min_f, p_min_b, X_n, E_n, G_n, &
    & v, j, Ecut, epsilon, naccept, nreject, tot_accept, tot_reject, &
    & valid, pot, constraint, n)
implicit none

integer, intent(in) :: n, v, j
double precision, intent(in) :: Ecut, epsilon
double precision, intent(inout) :: X_pls(n), p_pls_f(n), p_pls_b(n), &
    & X_min(n), p_min_f(n), p_min_b(n)
double precision, intent(out) :: X_n(n), E_n, G_n(n)
integer, intent(out) :: naccept, nreject, tot_accept, tot_reject
logical, intent(out) :: valid

external pot
!f2py intent(in), depend(n) :: X_pls
!f2py intent(hide) :: n
!f2py intent(out) :: E_n
!f2py intent(out), depend(n) :: G_n
external constraint
!f2py intent(in), depend(n) :: X_pls
!f2py intent(hide) :: n
!f2py intent(out) :: E_n2
!f2py intent(out), depend(n) :: G_n2

logical accept, valid2, Eaccept, Caccept
double precision X_n2(n), E_n2, G_n2(n), prob_new, L2, M2, N2
double precision X_dummy(n), p_dummy_f(n), p_dummy_b(n)
integer naccept2, nreject2, tot_accept2, tot_reject2, k

if (j.eq.0) then
    ! Base case: Take a single leapfrog step in the direction v.
    ! Because pot and constraint need to be called within this subroutine
    ! explicitly include subroutine takestep
    if (v.eq.-1) then
        ! take step
        X_min(1:n) = X_min(1:n) - p_min_b(1:n) * epsilon
        ! calculate potential and constraint
        call pot(X_min, E_n, G_n, n)
        call constraint(X_min, E_n2, G_n2, n)
        ! reflect off potential/constraint if new point not within contour
        Eaccept = Ecut.ge.E_n
        Caccept = 0.d0.ge.E_n2
        if (Eaccept.and.Caccept) then
            ! do nothing
            accept = .TRUE.
        else if (Eaccept.and.(.not.Caccept)) then
            ! reflect off constraint gradient
            accept = .FALSE.
            L2 = 0.d0
            do k=1,n
                L2 = L2 + G_n2(k)**2
            end do
            p_min_f(1:n) = p_min_b(1:n)
            p_min_b(1:n) = p_min_b(1:n) &
                & - 2.d0 * G_n2(1:n) * dot_product(G_n2(1:n), p_min_b(1:n)) / L2
        else if(Caccept.and.(.not.Eaccept)) then
            ! reflect off gradient
            accept = .FALSE.
            do k=1,n
                L2 = L2 + G_n(k)**2
            end do
            p_min_f(1:n) = p_min_b(1:n)
            p_min_b(1:n) = p_min_b(1:n) &
                & - 2.d0 * G_n(1:n) * dot_product(G_n(1:n), p_min_b(1:n)) / L2
        else if((.not.Eaccept).and.(.not.Caccept)) then
            ! reflect of sum of normed gradient and constraint gradient
            accept = .FALSE.
            L2 = 0.d0
            M2 = 0.d0
            do k=1,n
                L2 = L2 + G_n(k)**2
                M2 = M2 + G_n2(k)**2
            end do
            L2 = sqrt(L2)
            M2 = sqrt(M2)
            N2 = 0.d0
            do k=1,n
                G_n2(k) = G_n2(k)/M2 + G_n(k)/L2
                N2 = N2 + G_n2(k)**2
            end do
            p_min_f(1:n) = p_min_b(1:n)
            p_min_b(1:n) = p_min_b(1:n) &
                & - 2.d0 * G_n2(1:n) * dot_product(G_n2(1:n), p_min_b(1:n)) / N2
        end if
        X_pls(1:n) = X_min(1:n)
        X_n(1:n) = X_min(1:n)
        p_pls_f(1:n) = p_min_f(1:n)
        p_pls_b(1:n) = p_min_b(1:n)
    else
        ! Assuming v==1
        ! take step
        X_pls(1:n) = X_pls(1:n) + p_pls_f(1:n) * epsilon
        ! calculate potential / constraint
        call pot(X_pls, E_n, G_n, n)
        call constraint(X_pls, E_n2, G_n2, n)
        ! check whether point is inside contour
        Eaccept = Ecut.ge.E_n
        Caccept = 0.d0.ge.E_n2
        ! reflect (or not)
        if (Eaccept.and.Caccept) then
            accept = .TRUE.
        else if (Eaccept.and.(.not.Caccept)) then
            ! reflect off constraint gradient
            accept = .FALSE.
            L2 = 0.d0
            do k=1,n
                L2 = L2 + G_n2(k)**2
            end do
            p_pls_b(1:n) = p_pls_f(1:n)
            p_pls_f(1:n) = p_pls_f(1:n) &
                & - 2.d0 * G_n2(1:n) * dot_product(G_n2(1:n), p_pls_f(1:n)) / L2
        else if(Caccept.and.(.not.Eaccept)) then
            ! reflect off gradient
            accept = .FALSE.
            L2 = 0.d0
            do k=1,n
                L2 = L2 + G_n(k)**2
            end do
            p_pls_b(1:n) = p_pls_f(1:n)
            p_pls_f(1:n) = p_pls_f(1:n) &
                & - 2.d0 * G_n(1:n) * dot_product(G_n(1:n), p_pls_f(1:n)) / L2
        else if((.not.Eaccept).and.(.not.Caccept)) then
            ! reflect of sum of normed gradient and constraint gradient
            accept = .FALSE.
            L2 = 0.d0
            M2 = 0.d0
            do k=1,n
                L2 = L2 + G_n(k)**2
                M2 = M2 + G_n2(k)**2
            end do
            L2 = sqrt(L2)
            M2 = sqrt(M2)
            N2 = 0.d0
            do k=1,n
                G_n2(k) = G_n2(k)/M2 + G_n(k)/L2
                N2 = N2 + G_n2(k)**2
            end do
            p_pls_b(1:n) = p_pls_f(1:n)
            p_pls_f(1:n) = p_pls_f(1:n) &
                & - 2.d0 * G_n2(1:n) * dot_product(G_n2(1:n), p_pls_f(1:n)) / N2
        end if
        X_min(1:n) = X_pls(1:n)
        X_n(1:n) = X_pls(1:n)
        p_min_b(1:n) = p_pls_f(1:n)
        p_min_b(1:n) = p_pls_b(1:n)
    end if

    if (accept) then
        naccept = 1
        nreject = 0
        tot_accept = 1
        tot_reject = 0
    else
        naccept = 0
        nreject = 1
        tot_accept = 0
        tot_reject = 1
    end if
    valid = .TRUE.
else
    ! Recursion: Implicitly build the height j-1 left and right subtrees.
    call build_tree( &
        & X_pls, p_pls_f, p_pls_b, X_min, p_min_f, p_min_b, X_n, E_n, G_n, &
        & v, j - 1, Ecut, epsilon, naccept, nreject, tot_accept, tot_reject, &
        & valid, pot, constraint, n)
    ! No need to keep going if the stopping criteria were met in
    ! the first subtree.
    if (valid) then
        if (v.eq.-1) then
            call build_tree( &
                & X_dummy, p_dummy_f, p_dummy_b, X_min, p_min_f, p_min_b, &
                & X_n2, E_n2, G_n2, v, j - 1, Ecut, epsilon, naccept2, &
                & nreject2, tot_accept2, tot_reject2, valid2, pot, constraint, n)
        else
            call build_tree( &
                & X_pls, p_pls_f, p_pls_b, X_dummy, p_dummy_f, p_dummy_b, &
                & X_n2, E_n2, G_n2, v, j - 1, Ecut, epsilon, naccept2, &
                & nreject2, tot_accept2, tot_reject2, valid2, pot, constraint, n)
        end if
        if (valid2) then
            if (naccept2.gt.0) then
                prob_new = dble(naccept2) / max(dble(naccept + naccept2), 1.d0)
                if ((naccept.eq.0).or.(r_uniform().lt.prob_new)) then
                    X_n(1:n) = X_n2(1:n)
                    G_n(1:n) = G_n2(1:n)
                    E_n = E_n2
                end if
                naccept = naccept + naccept2
            end if
            nreject = nreject + nreject2
            valid = stop_criterion(X_pls, p_pls_f, X_min, p_min_b, n)
        else
            valid = .FALSE.
        end if
        tot_accept = tot_accept + tot_accept2
        tot_reject = tot_reject + tot_reject2
    end if
end if

end subroutine

function stop_criterion(X_pls, p_pls_f, X_min, p_min_b, n) result(criterion)
implicit none

integer n
double precision X_pls(n), p_pls_f(n), X_min(n), p_min_b(n)
logical criterion

integer j
double precision diff(n), linear(3), rel_p, p_pls(n), p_min(n)


diff(1:n) = X_pls(1:n) - X_min(1:n)

if (remove_angular_momentum) then
    p_pls(:) = zero_angular_momentum(X_pls(:), p_pls_f(:), n)
    p_min(:) = zero_angular_momentum(X_min(:), p_min_b(:), n)
else
    p_pls(:) = p_pls_f(:)
    p_min(:) = p_min_b(:)
end if

if (remove_linear_momentum) then
    linear(:) = 0.d0
    do j=1,n-2,3
        linear(:) = p_pls(j:j+2)
    end do
    linear(:) = linear(:)/dble(n)
    rel_p = 0.d0
    do j=1,n-2,3
        rel_p = rel_p + dot_product(diff(j:j+2), linear(:))
    end do
    ! Only calculates condition for negative velocity if
    ! the condition for the forward momentum is satified
    if (dot_product(diff, p_pls).ge.rel_p) then
        linear(:) = 0.d0
        do j=1,n-2,3
            linear(:) = p_min(j:j+2)
        end do
        linear(:) = linear(:)/dble(n)
        rel_p = 0.d0
        do j=1,n-2,3
            rel_p = rel_p + dot_product(diff(j:j+2), linear(:))
        end do

        if (dot_product(diff, p_min).ge.rel_p) then
            criterion = .TRUE.
        else
            criterion = .FALSE.
        end if
    else
        criterion = .FALSE.
    end if
else
    criterion = (dot_product(diff, p_pls).ge.0.d0).and.(dot_product(diff, p_min).ge.0.d0)
end if

end function

function zero_angular_momentum(coords, p, n) result(new_p)
implicit none

integer n
double precision coords(n), p(n), new_p(n)

integer j
double precision I(3,3), omega(3), R0(3), L(3)
double precision X, Y, Z, X2, XY, ZX, Y2, YZ, Z2

! Finding centre of mass
R0(:) = 0.d0
do j=1,n-2,3
    R0(:) = R0(:) + coords(j:j+2)
end do
R0(:) = R0(:)/dble(n/3)

X2 = 0.d0
XY = 0.d0
ZX = 0.d0
Y2 = 0.d0
YZ = 0.d0
Z2 = 0.d0
L(:) = 0.d0
! save zeroed coordinates in new_p vector
do j=1,n-2,3
    new_p(j:j+2) = coords(j:j+2) - R0(:)
    L(:) = L(:) + cross(new_p(j:j+2), p(j:j+2))
    X2 = X2 + new_p(j)*new_p(j)
    XY = XY + new_p(j)*new_p(j+1)
    ZX = ZX + new_p(j+2)*new_p(j)
    Y2 = Y2 + new_p(j+1)*new_p(j+1)
    YZ = YZ + new_p(j+1)*new_p(j+2)
    Z2 = Z2 + new_p(j+2)*new_p(j+2)
end do

I(1,1:3) = (/Y2 + Z2, -XY, -ZX/)
I(2,1:3) = (/-XY, X2 + Z2, -YZ/)
I(3,1:3) = (/-ZX, -YZ, X2 + Y2/)
I(:,:) = matinv3(I(:,:))
omega(:) = matmul(I(:,:), L(:))

do j=1,n-2,3
    new_p(j:j+2) = p(j:j+2) - cross(omega(:), new_p(j:j+2))
end do

end function

function cross(a, b)
implicit none
double precision cross(3)
double precision  a(3), b(3)

cross(1) = a(2) * b(3) - a(3) * b(2)
cross(2) = a(3) * b(1) - a(1) * b(3)
cross(3) = a(1) * b(2) - a(2) * b(1)
end function

function matinv3(A) result(B)
  implicit none
  !! Performs a direct calculation of the inverse of a 3×3 matrix.
  double precision A(3,3), B(3,3), detinv

  ! Calculate the inverse determinant of the matrix
  detinv = 1/(A(1,1)*A(2,2)*A(3,3) - A(1,1)*A(2,3)*A(3,2)&
            - A(1,2)*A(2,1)*A(3,3) + A(1,2)*A(2,3)*A(3,1)&
            + A(1,3)*A(2,1)*A(3,2) - A(1,3)*A(2,2)*A(3,1))

  ! Calculate the inverse of the matrix
  B(1,1) = +detinv * (A(2,2)*A(3,3) - A(2,3)*A(3,2))
  B(2,1) = -detinv * (A(2,1)*A(3,3) - A(2,3)*A(3,1))
  B(3,1) = +detinv * (A(2,1)*A(3,2) - A(2,2)*A(3,1))
  B(1,2) = -detinv * (A(1,2)*A(3,3) - A(1,3)*A(3,2))
  B(2,2) = +detinv * (A(1,1)*A(3,3) - A(1,3)*A(3,1))
  B(3,2) = -detinv * (A(1,1)*A(3,2) - A(1,2)*A(3,1))
  B(1,3) = +detinv * (A(1,2)*A(2,3) - A(1,3)*A(2,2))
  B(2,3) = -detinv * (A(1,1)*A(2,3) - A(1,3)*A(2,1))
  B(3,3) = +detinv * (A(1,1)*A(2,2) - A(1,2)*A(2,1))
end function

end module
