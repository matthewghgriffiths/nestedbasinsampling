



subroutine hardshellenergy(coords, natoms, radius, e)
implicit none
integer, intent(in) :: natoms
double precision, intent(in) :: coords(3*natoms), radius
double precision, intent(out) :: e

double precision dr(3), r2, r2cut
integer j

r2cut = radius**2

e = 0.d0
do j=1, natoms
    dr(:) = coords(3*(j-1)+1 : 3*(j-1) + 3)
    r2 = sum(dr(:)**2)
    if (r2 > r2cut) e = e + (r2 - r2cut)/2 !sqrt(r2) - radius
enddo

end subroutine hardshellenergy

subroutine hardshellenergy_gradient(coords, natoms, radius, e, grad)
implicit none
integer, intent(in) :: natoms
double precision, intent(in) :: coords(3*natoms), radius
double precision, intent(out) :: e, grad(3*natoms)

double precision r, r2, r2cut
integer j

r2cut = radius**2

e = 0.d0
grad(:) = 0.d0
do j=1,3*natoms-2,3
    r2 = dot_product(coords(j:j+2),coords(j:j+2)) 
    if (r2 > r2cut) then
        e = e + (r2 - r2cut)/2
        grad(j:j+2) = coords(j:j+2)
    endif
enddo

end subroutine hardshellenergy_gradient
