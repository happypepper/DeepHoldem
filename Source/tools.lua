--- Assorted tools.
--@module tools
local M = {C = {}, max_choose = 55}

--- Generates a string representation of a table.
--@param table the table
--@return the string
function M:table_to_string(table)
  local out = "{"
  for key,value in pairs(table) do

    local val_string = ''

    if type(value) == 'table' then
      val_string = self:table_to_string(value)
    else
      val_string = tostring(value)
    end

    out = out .. tostring(key) .. ":" .. val_string .. ", "
  end

  out = out .. "}"
  return out
end

--- An arbitrarily large number used for clamping regrets.
--@return the number
function M:max_number()
  return 999999
end

--- Initializes the choose table.
-- @local
function M:_init_choose()
  for i = 0,self.max_choose do
    for j = 0,self.max_choose do
      self.C[i*self.max_choose + j] = 0
    end
  end

  for i = 0,self.max_choose do
    self.C[i*self.max_choose] = 1
    self.C[i*self.max_choose + i] = 1
  end

  for i = 1,self.max_choose do
    for j = 1,i do
      self.C[i*self.max_choose + j] = self.C[(i-1)*self.max_choose + j-1] + self.C[(i-1)*self.max_choose + j]
    end
  end
end

M:_init_choose()

function M:choose(n, k)
  return self.C[n*self.max_choose + k]
end

return M
