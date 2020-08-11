module MyFunc
    export printTable

    using PrettyTables

    function printTable(correct::Vector{Int}, discrimination::Vector{Int})
        table = zeros((3, 3))
        for i in 1:size(correct)[1]
            c = correct[i]
            d = discrimination[i]
            table[c, d] += 1
        end
        pretty_table(table)
    end
end